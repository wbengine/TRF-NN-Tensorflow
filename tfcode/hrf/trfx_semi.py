import numpy as np
import time
from collections import OrderedDict

from base import seq, log
from . import trfx
from .trfx import DefaultOps


class Config(trfx.Config):
    def __init__(self, data):
        super().__init__(data)

        self.train_batch_size = 1000
        self.full_batch_size = 100
        self.inter_alpha = 100


class TRF(trfx.TRF):
    def __init__(self, config, data_x, data_full,
                 logdir, device='/gpu:0', name='trf'):
        super().__init__(config, data_x, logdir, device, name)

        self.data_full = data_full
        self.data_x = data_x

    def update(self, data_list, sample_list, data_full_list=None):
        if data_full_list is None:
            return super().update(data_list, sample_list)

        # compute the scalars
        data_scalar = np.ones(len(data_list)) / len(data_list)
        sample_len = np.array([len(x) for x in sample_list])
        sample_facter = np.array(self.config.pi_true[self.config.min_len:]) / \
                        np.array(self.config.pi_0[self.config.min_len:])
        sample_scalar = sample_facter[sample_len - self.config.min_len] / len(sample_list)

        # update word phi
        if not self.config.fix_trf_model:
            with self.time_recoder.recode('update_word'):
                self.phi_word.update(data_list, data_scalar, sample_list, sample_scalar,
                                     learning_rate=self.cur_lr_word)

        if not self.config.fix_crf_model:

            data_full_scalar = self.config.inter_alpha * np.ones(len(data_full_list)) / len(data_full_list)

            data_part_list = data_list + sample_list + data_full_list
            data_part_scalar = -np.concatenate([data_scalar, -sample_scalar, -data_full_scalar], axis=0)

            # forward-backward for data
            data_part_list_x = [s.x[0] for s in data_part_list]
            with self.time_recoder.recode('update_marginal_data'):
                data_fp_logps_list, logzs_data = self.marginal_logps(data_part_list_x)

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
            self.norm_const.update(sample_list, learning_rate=self.cur_lr_logz)
            logz1 = self.get_true_logz(self.config.min_len)[0]
            self.norm_const.set_logz1(logz1)

        # update simulater
        with self.time_recoder.recode('update_simulater'):
            self.sampler.update(seq.get_x(sample_list))

        # update dbg info
        self.sample_cur_pi.fill(0)
        for x in sample_list:
            self.sample_cur_pi[len(x)] += 1
        self.sample_acc_count += self.sample_cur_pi
        self.sample_cur_pi /= self.sample_cur_pi.sum()

        dbg_info = dict()
        dbg_info['logz1'] = logz1
        acc_pi = self.sample_acc_count / np.sum(self.sample_acc_count)
        dbg_info['pi_dist'] = np.arccos(np.dot(acc_pi, self.config.pi_0) /
                                        np.linalg.norm(acc_pi) / np.linalg.norm(self.config.pi_0))

        return dbg_info

    def train(self, print_per_epoch=0.1, operation=None):

        # initialize
        self.initialize()

        if self.exist_model():
            self.restore()
        if self.config.load_crf_model is not None:
            self.restore_crf(self.config.load_crf_model)
        if self.config.load_trf_model is not None:
            self.restore_trf(self.config.load_trf_model)

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

            # draw samples
            with self.time_recoder.recode('sample'):
                sample_seqs = self.draw(self.config.sample_batch_size)

            # update paramters
            with self.time_recoder.recode('update'):
                # learining rate
                self.cur_lr_word = self.config.lr_word.get_lr(step+1, epoch)
                self.cur_lr_tag = self.config.lr_tag.get_lr(step+1, epoch)
                self.cur_lr_mix = self.config.lr_mix.get_lr(step+1, epoch)
                self.cur_lr_logz = self.config.lr_logz.get_lr(step+1, epoch)
                # update
                update_info = self.update(data_seqs, sample_seqs, data_full_seqs)

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
                info['lj_rate'] = self.sampler.lj_rate
                info['mv_rate'] = self.sampler.mv_rate
                info['logz1'] = self.update_global_norm()
                info.update(update_info)
                info['train'] = np.mean(model_train_nll[-epoch_step_num:])
                # info['train_phi'] = np.mean(model_train_nll_phi[-100:])
                # info['valid'] = model_valid_nll
                # info['auxil'] = np.mean(model_q_nll[-epoch_step_num:])
                # info['kl_dist'] = np.mean(model_kl_dist[-epoch_step_num:])

                x_list = seq.get_x(sample_seqs)
                info['kl_dist'] = np.mean(-self.get_logpxs(x_list, for_eval=False)) - self.sampler.eval(x_list)[0]

                ##########
                true_logz = None
                if self.config.max_len <= 5:
                    true_logz = np.array(self.get_true_logz())
                    sa_logz = np.array(self.norm_const.get_logz())
                    self.norm_const.set_logz(true_logz)
                    true_nll_train = self.eval(train_list)[0]
                    self.norm_const.set_logz(sa_logz)

                    info['true_train'] = true_nll_train

                log.print_line(info)

                print('[end]')
                # self.debug_logz()

                # write time
                f = self.write_files.get('time')
                f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, time_since_beg))
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.time_recoder.items()]) + '\n')
                f.flush()

                #  write zeta, logz, pi
                self.write_log_zeta(step, true_logz)