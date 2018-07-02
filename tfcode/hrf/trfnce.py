# using trf MCMC method to generate word sequences

import tensorflow as tf
import os
import json
import time
from copy import deepcopy
from collections import OrderedDict
from base import *
from lm import *

from trf.common import feat2 as feat
from trf.isample import sampler
from trf.nce import pot as nce_pot
from . import crf, pot


class Config(crf.Config):
    def __init__(self, data):
        super().__init__(data)

        Config.value_encoding_map[lr.LearningRate] = str

        self.min_len = data.get_min_len()
        self.max_len = data.get_max_len()
        self.pi_true = data.get_pi_true()
        self.pi_0 = self.pi_true
        self.word_vocab_size = data.get_vocab_size()
        self.tag_vocab_size = data.get_tag_size()
        self.beg_tokens = data.get_beg_tokens()  # [word_beg_token, tag_beg_token]
        self.end_tokens = data.get_end_tokens()  # [word_end_token, tag_end_token]

        # prior model path
        self.prior_model_path = None

        # for words
        self.word_config = None  # features

        # init zeta
        self.norm_type = 'linear'  # 'linear', or, 'multiple'
        self.init_logz = [0, np.log(self.word_vocab_size) + np.log(self.tag_vocab_size)]

        # init CRF
        self.load_crf_model = None
        self.fix_crf_model = False

        self.load_trf_model = None
        self.fix_trf_model = False

        # NCE
        self.batch_size = 100
        self.noise_factor = 1
        self.data_factor = 1  # the generated data rate
        self.sampler_config = sampler.LSTM.Config(self.word_vocab_size, 200, 1)
        self.sampler_config.learning_rate = 0.1
        self.word_average = False
        self.semi_supervised = False
        self.update_aux_using_samples = False

        # learning rate
        self.lr_word = lr.LearningRateTime(1e-3)
        self.lr_logz = lr.LearningRateTime(1.0, 0.2)
        self.opt_word = 'adam'
        self.opt_logz = 'sgd'
        self.zeta_gap = 10
        self.max_epoch = 1000

        # dbg
        self.write_dbg = False

    @property
    def beg_token(self):
        return self.beg_tokens[0]

    @property
    def end_token(self):
        return self.end_tokens[0]

    def create_feat_config(self, feat_files):
        feat_dict_all = feat.read_feattype_file(feat_files)

        tag_feat = dict()
        mix_feat = dict()
        wod_feat = dict()

        for ftype, cutoff in feat_dict_all.items():
            if ftype.find('w') != -1 and ftype.find('c') != -1:
                mix_feat[ftype] = cutoff
            elif ftype.find('w') != -1:
                wod_feat[ftype] = cutoff
            elif ftype.find('c') != -1:
                tag_feat[ftype] = cutoff

        self.tag_config.feat_dict = tag_feat
        self.tag_config.L2_reg = 0
        self.mix_config.feat_dict = mix_feat
        self.mix_config.L2_reg = 0
        if len(wod_feat) > 0:
            self.word_config = pot.FeatConfig()
            self.word_config.feat_dict = wod_feat
            self.word_config.L2_reg = 1e-6
        else:
            self.word_config = None

    def get_initial_logz(self, c=None):
        if c is None:
            c = np.log(self.word_vocab_size) + np.log(self.tag_vocab_size)

        len_num = self.max_len - self.min_len + 1
        logz = c * (np.linspace(1, len_num, len_num))
        return logz

    def __str__(self):
        s = 'trf_nce_noise{}_data{}'.format(self.noise_factor, self.data_factor)
        if self.prior_model_path is not None:
            s += '_priorlm'

        if self.word_config is not None:
            s += '_' + str(self.word_config)

        if self.tag_config is not None:
            s += '_' + str(self.tag_config)

        if self.mix_config is not None:
            s += '_' + str(self.mix_config)

        if self.load_crf_model is not None:
            if self.fix_crf_model:
                s += '_LoadCRFFix'
            else:
                s += '_LoadCRF'

        if self.load_trf_model is not None:
            if self.fix_trf_model:
                s += '_LoadTRFFix'
            else:
                s += '_LoadTRF'
        return s


class TRF(crf.CRF):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):

        super().__init__(config, data, logdir, device, name)

        # phi for words
        self.phi_word = pot.create(config.word_config, data.datas[0], config.opt_word, device)

        # logZ
        # self.norm_const = pot.Norm(config, data, config.opt_logz_method)
        if self.config.norm_type == 'linear':
            self.norm_const = nce_pot.NormLinear(config, data, opt_method=config.opt_logz)
        elif self.config.norm_type == 'multiple':
            self.norm_const = nce_pot.Norm(config, data, opt_method=config.opt_logz)
        else:
            raise TypeError('undefined norm_config = ', self.config.norm_config)

        # sampler
        self.sampler = sampler.LSTM(config, device)
        self.sampler.len_prob = config.pi_0
        self.sampler.update_len_prob = False

        # learning rate
        self.cur_lr_word = 1.0
        self.cur_lr_logz = 1.0

    def save(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        super().save(fname)
        self.phi_word.save(fname)
        self.norm_const.save(fname + '.norm')

    def restore(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        super().restore(fname)
        self.phi_word.restore(fname)
        self.norm_const.restore(fname + '.norm')

    def restore_crf(self, fname):
        super().restore(fname)

    def restore_trf(self, fname):
        print('[HRF] restore TRF from', fname)
        self.phi_word.restore(fname)

    @property
    def session(self):
        return tf.get_default_session()

    def phi(self, seq_list, filter_name='all'):
        """
        Args:
            seq_list: a list of Seq()
            filter_name: str,
                'all': all the phi
                'word': all the phi only depending on word
                'tag': all the phi only depending on tag

        Returns:

        """
        if filter_name == 'all':
            return super().phi(seq_list) + \
                   self.phi_word.get_value(seq_list)
        elif filter_name == 'word':
            return self.phi_word.get_value(seq_list)
        elif filter_name == 'tag':
            return self.phi_tag.get_value(seq_list)
        else:
            raise TypeError('[trf.phi] undefined filter name = ' + filter_name)

    def normalize(self, logp, lengths, for_eval=True):
        if for_eval:
            # using pi_true
            logp_m = logp + np.log(self.config.pi_true[lengths]) - self.norm_const.get_logz(lengths)
        else:
            # using pi_0
            logp_m = logp + np.log(self.config.pi_0[lengths]) - self.norm_const.get_logz(lengths)

        return logp_m

    def logps(self, seq_list, for_eval=True):
        """
        compute the logprobs of a list of Seq()
        Args:
            seq_list: a list of Seq()
            for_eval: if Ture, use the pi_true, else use pi_0

        Returns:
            a np.arrdy()
        """
        phi = self.phi(seq_list)

        seq_lens = np.array([len(s) for s in seq_list])
        # if np.any(seq_lens < self.config.min_len) or np.any(seq_lens > self.config.max_len):
        #     raise TypeError('min_len={}, max_len={} lens={}'.format(min(seq_lens), max(seq_lens), seq_lens))

        return self.normalize(phi, seq_lens, for_eval)

    def get_log_probs(self, seq_list, is_norm=True, for_eval=True, batch_size=100):
        logps = np.zeros(len(seq_list))

        for i in range(0, len(seq_list), batch_size):
            if is_norm:
                logps[i: i+batch_size] = self.logps(seq_list[i: i+batch_size], for_eval)
            else:
                logps[i: i+batch_size] = self.phi(seq_list[i: i+batch_size])

        return logps

    def rescore(self, x_list):

        # tag_list, _ = self.get_tag(x_list)
        # seq_list = [seq.Seq([x, t]) for x, t in zip(x_list, tag_list)]
        # return -self.get_log_probs(seq_list)

        return -self.get_logpxs(x_list)

    def get_logpxs(self, x_list, is_norm=True, for_eval=True, batch_size=100):
        logpx = np.zeros(len(x_list))

        for i in range(0, len(x_list), batch_size):
            logpx[i: i+batch_size] = self.logpxs(x_list[i: i+batch_size], is_norm, for_eval)
        return logpx

    def logpxs(self, x_list, is_norm=True, for_eval=True, logz_list=None):

        if logz_list is not None:
            logw_h = np.array(logz_list)  # get from super().marginal_logps(...)
        else:
            logw_h = super().logz(x_list)

        seq_list = [seq.Seq(x) for x in x_list]
        seq_lens = [len(x) for x in x_list]
        logw_x = self.phi(seq_list, 'word')

        if is_norm:
            return self.normalize(logw_x + logw_h, seq_lens, for_eval)
        else:
            return logw_x + logw_h

    def get_true_logz(self, max_len=None):
        if max_len is None:
            max_len = self.config.max_len

        logz = np.zeros(max_len - self.config.min_len + 1)
        for l in range(self.config.min_len, max_len + 1):
            x_list = []
            for x in sp.SeqIter(l, self.config.word_vocab_size,
                                beg_token=self.config.beg_tokens[0],
                                end_token=self.config.end_tokens[0]):
                x_list.append(x)
            logpx = self.get_logpxs(x_list, False)
            logz[l-self.config.min_len] = logsumexp(logpx)

        return logz

    def eval(self, seq_list, for_eval=True, type='word', is_norm=True):
        if type == 'joint':
            logps = self.get_log_probs(seq_list, is_norm=is_norm, for_eval=for_eval)
        else:
            logps = self.get_logpxs(seq.get_x(seq_list), is_norm=is_norm, for_eval=for_eval)
        nll = -np.mean(logps)
        words = np.sum([len(x)-1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)

        return nll, ppl

    def cmp_cluster_logps(self, logpm, logpn):
        logpc = logpm - np.logaddexp(logpm, np.log(self.config.noise_factor) + logpn)
        return logpc

    def cmp_cluster_1mlogps(self, logpm, logpn):
        logpc = np.log(self.config.noise_factor) + logpn - np.logaddexp(logpm, np.log(self.config.noise_factor) + logpn)
        return logpc

    def cmp_words(self, lengths, data_num):
        word_d = np.sum(lengths[0: data_num]) - data_num
        word_n = np.sum(lengths[data_num:]) - (len(lengths) - data_num)
        return word_d, word_n

    def cmp_cluster_weight(self, logpm, logpn, data_num, lengths):
        """
            w(x) = (1 - P(C=0|x)) / W_D, if x in data
            w(x) = - P(C=0|x) / (W_B/\nu),     if x in noise
        Args:
            logpm:  p_m
            logpn:  p_n
            data_num: data number
            lengths: the length of each sequence

        Returns:
            w(x) for x in {data, noise}
        """

        logpc = self.cmp_cluster_logps(logpm, logpn)
        if self.config.word_average:
            word_d, word_n = self.cmp_words(lengths, data_num)
            w_d = (1 - np.exp(logpc[0: data_num])) / word_d
            w_n = - np.exp(logpc[data_num:]) / (word_n / self.config.noise_factor)
        else:
            w_d = (1 - np.exp(logpc[0: data_num])) / data_num
            w_n = - np.exp(logpc[data_num:]) / data_num
        w = np.concatenate([w_d, w_n])

        return w

    def cmp_cluster_m(self, logpm, logpn, data_num, lengths):
        """
            m(x) = 1/D * (\mu p_m p_n) / (p_m + \mu p_n)^2
        Args:
            logpm: p_m(x)
            logpn: p_n(x)
            data_num: the data num
            lengths: the length of each sequence

        Returns:
            m(x)
        """
        logm = np.log(self.config.noise_factor) + logpm + logpn - \
               2 * np.logaddexp(logpm, np.log(self.config.noise_factor) + logpn)

        if self.config.word_average:
            word_d, word_n = self.cmp_words(lengths, data_num)
            m_d = np.exp(logm[0:data_num]) / word_d
            m_n = np.exp(logm[data_num:]) / (word_n / self.config.noise_factor)
        else:
            m_d = np.exp(logm[0:data_num]) / data_num
            m_n = np.exp(logm[data_num:]) / data_num
        return np.concatenate([m_d, m_n])

    def cmp_cluster_loss(self, logpm, logpn, data_num, lengths):
        if self.config.word_average:
            word_d, word_n = self.cmp_words(lengths, data_num)
            loss_d = self.cmp_cluster_logps(logpm[0:data_num], logpn[0:data_num]) / word_d
            loss_n = self.cmp_cluster_1mlogps(logpm[data_num:], logpn[data_num:]) / (word_n / self.config.noise_factor)
        else:
            loss_d = self.cmp_cluster_logps(logpm[0:data_num], logpn[0:data_num]) / data_num
            loss_n = self.cmp_cluster_1mlogps(logpm[data_num:], logpn[data_num:]) / data_num
        return -np.concatenate([loss_d, loss_n])

    def initialize(self):
        # create features
        super().initialize()

        print('[TRF] init word features.')
        self.phi_word.initialize()

        # start sampler
        self.sampler.initialize()

        # print parameters
        print('[TRF] word_feat_num = {:,}'.format(self.phi_word.get_param_num()))

    def update_aux(self, seq_x_list, model_logps, noise_logps, data_num):

        sample_num = len(seq_x_list) - data_num

        scale_data = 1.0 / data_num * np.ones(data_num)

        # n = len(seq_x_list) - data_num
        # lengths = [len(x) for x in seq_x_list[data_num:]]
        logw = model_logps[data_num:] - noise_logps[data_num:]

        # logw_dict = dict()
        # for j, w in zip(lengths, logw):
        #     a = logw_dict.setdefault(j, [])
        #     a.append(w)
        #
        # logzl = np.zeros(self.config.max_len + 1)
        # for j, a in logw_dict.items():
        #     logzl[j] = logsumexp(a) - np.log(len(a))  # 1/n_l * sum_x^l e^phi(x^l) / q_l(x^l)
        #
        # log_scale = logw - logzl[lengths] - np.log(n)

        log_scale = logw - logsumexp(logw)
        scale_sample = np.exp(log_scale)

        scale = np.concatenate([scale_data, scale_sample]) * 0.5

        if self.config.update_aux_using_samples:
            # both
            self.sampler.update(seq_x_list, scale)
        else:
            # all training
            self.sampler.update(seq_x_list[0:data_num], scale_data)

        return scale

    def update(self, data_list):

        data_x_list = seq.get_x(data_list)
        src_data_num = len(data_list)

        # generate noise samples
        with self.time_recoder.recode('sampling'):
            data_x_list = data_x_list + self.sampler.generate(int(self.config.data_factor * len(data_x_list)))
            # data_list = self.sampler.add_noise(data_list)
            sample_x_list = self.sampler.generate(int(len(data_x_list) * self.config.noise_factor))

            seq_x_list = data_x_list + sample_x_list
            noise_logps = self.sampler.get_log_probs(seq_x_list)
            data_num = len(data_x_list)
            seq_lens = [len(x) for x in seq_x_list]

        # comput the marginal logps
        with self.time_recoder.recode('update_marginal'):
            fb_logps_list, logz_list = self.marginal_logps(seq_x_list)

        with self.time_recoder.recode('loss'):
            model_logps = self.logpxs(seq_x_list, for_eval=False, logz_list=logz_list)  # for training to calculate the logp
            cluster_weights = self.cmp_cluster_weight(model_logps, noise_logps, data_num, seq_lens)
            loss_all = self.cmp_cluster_loss(model_logps, noise_logps, data_num, seq_lens)
            loss = np.sum(loss_all)

        # set data and sample
        data_x_list = data_x_list[0: src_data_num]
        sample_list = [seq.Seq(x) for x in seq_x_list[src_data_num:]]
        sample_x_list = seq_x_list[src_data_num:]
        data_scalar = cluster_weights[0: src_data_num]
        sample_scalar = - cluster_weights[src_data_num:]

        # update word phi
        if not self.config.fix_trf_model:
            with self.time_recoder.recode('update_word'):
                self.phi_word.update(data_list, data_scalar, sample_list, sample_scalar,
                                     learning_rate=self.cur_lr_word)

        # if not self.config.fix_crf_model:
        if not self.config.fix_crf_model:

            if self.config.semi_supervised:
                data_fp_logps_list = fb_logps_list[0: src_data_num]
            else:
                data_fp_logps_list = None
            sample_fp_logps_list = fb_logps_list[src_data_num:]

            with self.time_recoder.recode('update_tag'):
                self.phi_tag.update(data_list, data_scalar, sample_list, sample_scalar,
                                    data_fp_logps_list=data_fp_logps_list,
                                    sample_fp_logps_list=sample_fp_logps_list,
                                    learning_rate=self.cur_lr_tag)

            with self.time_recoder.recode('update_mix'):
                self.phi_mix.update(data_list, data_scalar, sample_list, sample_scalar,
                                    data_fp_logps_list=data_fp_logps_list,
                                    sample_fp_logps_list=sample_fp_logps_list,
                                    learning_rate=self.cur_lr_mix)

        # update zeta
        with self.time_recoder.recode('update_logz'):
            self.norm_const.update(seq_x_list, cluster_weights, cluster_m=None,
                                   learning_rate=self.cur_lr_logz)
            # logz0 = self.get_true_logz(self.config.min_len)[0]

        # update simulater
        with self.time_recoder.recode('update_simulater'):
            update_aux_scale = self.update_aux(data_x_list, model_logps, noise_logps, src_data_num)
            # self.sampler.update(data_x_list[0: src_data_num], np.ones(src_data_num) / src_data_num)
            sampler_ll = self.sampler.eval_nll(data_x_list[0: src_data_num])

        # update dbg info
        if self.config.write_dbg:
            f = self.write_files.get('noise')
            f.write('step={}\n'.format(self.training_info['trained_step']))
            f.write('[d/s] [model_logp] [noise_logp] [cluster_w] [cluster_p] [ scale ] [ seq ]\n')
            for i, s in enumerate(seq_x_list):
                f.write('{:>5} {:<12.5f} {:<12.5f} {:<12.5f} {:<8.5f} {:<8.5f}'.format(
                    'd' if i < len(data_x_list) else 's',
                    model_logps[i], noise_logps[i], cluster_weights[i], np.exp(-loss_all[i]), update_aux_scale[i]))
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
        print_infos['ESR'] = np.sum(update_aux_scale >= 1e-4) / len(update_aux_scale)
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
        epoch_step_num = self.data.get_epoch_step_num()
        print('[TRF] epoch_step_num={}'.format(epoch_step_num))
        print('[TRF] train_list={}'.format(len(train_list)))
        print('[TRF] valid_list={}'.format(len(valid_list)))
        last_epoch = 0
        epoch = 0
        print_next_epoch = 0
        for step, data_seqs in enumerate(self.data):

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
                self.cur_lr_word = self.config.lr_word.get_lr(step+1, epoch)
                self.cur_lr_tag = self.config.lr_tag.get_lr(step+1, epoch)
                self.cur_lr_mix = self.config.lr_mix.get_lr(step+1, epoch)
                self.cur_lr_logz = self.config.lr_logz.get_lr(step+1, epoch)
                # update
                nce_loss, update_info = self.update(data_seqs)

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
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.time_recoder.items()]) + '\n')
                f.flush()


class DefaultOps(crf.DefaultOps):
    def __init__(self, m, nbest_or_nbest_file_tuple, tagging_test_list):

        super().__init__(m, tagging_test_list)

        if isinstance(nbest_or_nbest_file_tuple, tuple) or \
                isinstance(nbest_or_nbest_file_tuple, list):
            print('[%s.%s] input the nbest files.' % (__name__, self.__class__.__name__))
            self.nbest_cmp = reader.NBest(*nbest_or_nbest_file_tuple)
        else:
            print('[%s.%s] input nbest computer.' % (__name__, self.__class__.__name__))
            self.nbest_cmp = nbest_or_nbest_file_tuple

        self.write_models = wb.mkdir(os.path.join(self.m.logdir, 'trf_models'))

    def perform(self, step, epoch):
        super().perform(step, epoch)

        # resocring
        time_beg = time.time()
        self.nbest_cmp.lmscore = self.m.rescore(self.nbest_cmp.get_nbest_list(self.m.data))
        rescore_time = time.time() - time_beg

        # compute wer
        time_beg = time.time()
        wer = self.nbest_cmp.wer()
        wer_time = time.time() - time_beg

        wb.WriteScore(self.write_models + '/epoch%.2f' % epoch + '.lmscore', self.nbest_cmp.lmscore)
        print('epoch={:.2f} test_wer={:.2f} lmscale={} '
              'rescore_time={:.2f}, wer_time={:.2f}'.format(
               epoch, wer, self.nbest_cmp.lmscale,
               rescore_time / 60, wer_time / 60))

        res = wb.FRes(os.path.join(self.m.logdir, 'results_nbest_wer.log'))
        res_name = 'epoch%.2f' % epoch
        res.Add(res_name, ['lm-scale'], [self.nbest_cmp.lmscale])
        res.Add(res_name, ['wer'], [wer])



