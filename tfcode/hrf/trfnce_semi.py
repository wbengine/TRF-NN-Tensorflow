# using trf MCMC method to generate word sequences

import tensorflow as tf
import os
import json
import time
from copy import deepcopy
from collections import OrderedDict
from base import *
from lm import *

from . import trfnce
from . import pot
from .trfnce import DefaultOps


class Config(trfnce.Config):
    def __init__(self, data):
        super().__init__(data)

        self.word_config = pot.NetConfig(data)
        self.data_factor = 0.5
        self.noise_factor = 1.0

        self.inter_alpha = 10
        self.full_batch_size = 100  # batch_size for full data

    def __str__(self):

        s = super().__str__()

        if self.inter_alpha == int(self.inter_alpha):
            s += '_alpha%d' % self.inter_alpha
        else:
            s += '_alpha%.2f' % self.inter_alpha

        return s


class TRF(trfnce.TRF):
    def __init__(self, config, data, data_full, logdir,
                 device='/gpu:0', name='trf'):

        super().__init__(config, data, logdir, device, name)

        self.data_full = data_full

    def update(self, data_list, data_full_list):

        data_x_list = seq.get_x(data_list)
        src_data_num = len(data_list)

        # generate noise samples
        with self.time_recoder.recode('sampling'):

            nd = len(data_list)
            k1 = int(nd * self.config.data_factor)
            k2 = int((nd + k1) * self.config.noise_factor)

            samples_all = self.sampler.generate(k1 + k2)

            seq_x_list = data_x_list + samples_all
            seq_list = [seq.Seq(x) for x in seq_x_list]
            noise_logps = self.sampler.get_log_probs(seq_x_list)
            data_num = nd + k1
            noise_num = k2
            seq_lens = [len(x) for x in seq_x_list]

        # comput the marginal logps
        with self.time_recoder.recode('update_marginal'):
            fp_logps_list, logz_list = self.marginal_logps(seq_x_list)

        with self.time_recoder.recode('loss'):
            model_logps = self.logpxs(seq_x_list, for_eval=False,
                                      logz_list=logz_list)  # for training to calculate the logp
            cluster_weights = self.cmp_cluster_weight(model_logps, noise_logps, data_num, seq_lens)
            loss_all = self.cmp_cluster_loss(model_logps, noise_logps, data_num, seq_lens)
            loss = np.sum(loss_all)

        # set data and sample
        # data_x_list = data_x_list[0: src_data_num]
        # sample_list = [seq.Seq(x) for x in seq_x_list[src_data_num:]]
        # sample_x_list = seq_x_list[src_data_num:]
        # data_scalar = cluster_weights[0: src_data_num]
        # sample_scalar = - cluster_weights[src_data_num:]

        # update word phi
        if not self.config.fix_trf_model:
            with self.time_recoder.recode('update_word'):
                self.phi_word.update(seq_list[0: data_num], cluster_weights[0: data_num],
                                     seq_list[data_num:], -cluster_weights[data_num:],
                                     learning_rate=self.cur_lr_word)

        # if not self.config.fix_crf_model:
        if not self.config.fix_crf_model:

            fp_logps_list2, _ = self.marginal_logps(seq.get_x(data_full_list))

            data_full_scalar = self.config.inter_alpha * np.ones(len(data_full_list)) / len(data_full_list)
            data_part_list = seq_list + data_full_list
            data_part_scalar = -np.concatenate([cluster_weights, -data_full_scalar],
                                               axis=0)

            data_fp_logps_list = np.concatenate([fp_logps_list, fp_logps_list2], axis=0)

            with self.time_recoder.recode('update_tag'):
                self.phi_tag.update(data_full_list, data_full_scalar, data_part_list, data_part_scalar,
                                    data_fp_logps_list=None,
                                    sample_fp_logps_list=data_fp_logps_list,
                                    learning_rate=self.cur_lr_tag)

            with self.time_recoder.recode('update_mix'):
                self.phi_mix.update(data_full_list, data_full_scalar, data_part_list, data_part_scalar,
                                    data_fp_logps_list=None,
                                    sample_fp_logps_list=data_fp_logps_list,
                                    learning_rate=self.cur_lr_mix)

        # update zeta
        with self.time_recoder.recode('update_logz'):
            self.norm_const.update(seq_x_list, cluster_weights, cluster_m=None,
                                   learning_rate=self.cur_lr_logz)
            # logz0 = self.get_true_logz(self.config.min_len)[0]

        # update simulater
        with self.time_recoder.recode('update_simulater'):
            # update_aux_scale = self.update_aux(data_x_list, model_logps, noise_logps, src_data_num)
            self.sampler.update(data_x_list, np.ones(src_data_num) / src_data_num)
            sampler_ll = self.sampler.eval_nll(data_x_list)

        # update dbg info
        if self.config.write_dbg:
            f = self.write_files.get('noise')
            f.write('step={}\n'.format(self.training_info['trained_step']))
            f.write('[d/s] [model_logp] [noise_logp] [cluster_w] [cluster_p] [ seq ]\n')
            for i, s in enumerate(seq_x_list):
                f.write('{:>5} {:<12.5f} {:<12.5f} {:<12.5f} {:<8.5f} '.format(
                    'd' if i < len(data_x_list) else 's',
                    model_logps[i], noise_logps[i], cluster_weights[i], np.exp(-loss_all[i])))
                f.write('[' + ' '.join(str(w) for w in s) + ']\n')
            f.flush()

        f = self.write_files.get('logz')
        logz = self.norm_const.get_var()
        grad = self.norm_const.get_gradient(seq_x_list, cluster_weights)
        # vars = self.norm_const.get_variance(seq_list, cluster_m)
        log.write_array(f, logz, 'logz')
        log.write_array(f, grad, 'grad')
        # log.write_array(f, vars, 'vars')
        f.write('\n')
        f.flush()

        print_infos = OrderedDict()
        print_infos['sumw'] = np.sum(np.abs(cluster_weights))
        print_infos['logz0'] = self.norm_const.get_logz(self.config.min_len)
        print_infos['aux_nll'] = sampler_ll
        # print_infos['kl'] = np.mean(model_logps[data_num:] - noise_logps[data_num:])

        self.time_recoder.merge(self.phi_mix.time_recoder)

        return loss, print_infos

    def train(self, print_per_epoch=0.1, operation=None):

        # initialize
        self.initialize()

        if self.exist_model():
            self.restore()
        if self.config.load_crf_model is not None:
            self.restore_crf(self.config.load_crf_model)
        if self.config.load_trf_model is not None:
            self.restore_trf(self.config.load_trf_model)

        # reinit the logz
        self.norm_const.set_logz0(self.get_true_logz(self.config.min_len)[0])

        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]

        print('[TRF] [Train]...')
        time_beginning = time.time()
        model_train_nll = []
        # model_train_nll_phi = []
        # model_q_nll = []
        # model_kl_dist = []

        self.data.train_batch_size = self.config.train_batch_size
        self.data.is_shuffle = True
        self.data_full.train_batch_size = self.config.full_batch_size
        self.data_full.is_shuffle = True
        epoch_step_num = self.data.get_epoch_step_num()
        print('[TRF] epoch_step_num={}'.format(epoch_step_num))
        print('[TRF] train_list={}'.format(len(train_list)))
        print('[TRF] valid_list={}'.format(len(valid_list)))
        last_epoch = 0
        epoch = 0
        print_next_epoch = 0
        for step, (data_seqs, data_full_seqs) in enumerate(zip(self.data, self.data_full)):

            ###########################
            # extra operations
            ###########################
            if operation is not None:
                operation.run(step, epoch)

            if int(self.data.get_cur_epoch()) > last_epoch:
                self.save()
                last_epoch = int(self.data.get_cur_epoch())

            if epoch >= self.config.max_epoch:
                print('[TRF] train stop!')
                self.save()
                # operation.perform(step, epoch)
                break

            # update epoches
            epoch = self.data.get_cur_epoch()

            # update training information
            self.training_info['trained_step'] += 1
            self.training_info['trained_epoch'] = self.data.get_cur_epoch()
            self.training_info['trained_time'] = (time.time() - time_beginning) / 60

            # update paramters
            with self.time_recoder.recode('update'):
                # learining rate
                self.cur_lr_word = self.config.lr_word.get_lr(step + 1, epoch)
                self.cur_lr_tag = self.config.lr_tag.get_lr(step + 1, epoch)
                self.cur_lr_mix = self.config.lr_mix.get_lr(step + 1, epoch)
                self.cur_lr_logz = self.config.lr_logz.get_lr(step + 1, epoch)
                # update
                nce_loss, update_info = self.update(data_seqs, data_full_seqs)

            # evaulate the nll
            with self.time_recoder.recode('eval_train_nll'):
                nll_train = self.eval(data_seqs)[0]
                model_train_nll.append(nll_train)
                # model_train_nll_phi.append(self.eval(data_seqs, is_norm=False)[0])
                # model_kl_dist.append(self.eval(sample_seqs)[0] - self.mcmc.eval(sample_seqs)[0])

            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                time_since_beg = (time.time() - time_beginning) / 60

                # with self.time_recoder.recode('eval'):
                #     model_valid_nll = self.eval(valid_list)[0]

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                info['lr_tag'] = '{:.2e}'.format(self.cur_lr_tag)
                info['lr_mix'] = '{:.2e}'.format(self.cur_lr_mix)
                info['lr_word'] = '{:.2e}'.format(self.cur_lr_word)
                info['lr_logz'] = '{:.2e}'.format(self.cur_lr_logz)
                info['loss'] = nce_loss
                info.update(update_info)
                info['train'] = np.mean(model_train_nll[-epoch_step_num:])
                # info['train_phi'] = np.mean(model_train_nll_phi[-100:])
                # info['valid'] = model_valid_nll

                log.print_line(info)
                print('[end]')

                # write time
                f = self.write_files.get('time')
                f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, time_since_beg))
                f.write(
                    ' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.time_recoder.items()]) + '\n')
                f.flush()



