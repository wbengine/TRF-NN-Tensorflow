import tensorflow as tf
import time
import json
import os
from collections import OrderedDict

from base import *
from trf.common import *
from trf.sa import simulater, pot
from trf.nce import noise
from trf.sa.trf import DefaultOps
from trf.nce.pot import NormLinear

from . import lstm, norm, sampler


class Config(wb.Config):
    def __init__(self, data):
        Config.value_encoding_map[lr.LearningRate] = str

        self.min_len = data.get_min_len()
        self.max_len = data.get_max_len()
        self.vocab_size = data.get_vocab_size()
        self.pi_true = data.get_pi_true()
        self.pi_0 = data.get_pi0(self.pi_true)
        self.beg_token = data.get_beg_token()
        self.end_token = data.get_end_token()

        self.global_norm = False

        # prior model path
        self.prior_model_path = None

        # for discrete features
        self.feat_config = pot.FeatConfig(data)

        # for network features
        self.net_config = pot.NetConfig(data)

        # init zeta
        self.norm_type = 'multiple'
        self.init_logz = self.get_initial_logz()

        # AugSA
        self.train_batch_size = 1000
        self.sample_batch_size = 100

        self.sampler_config = sampler.LSTM.Config(self.vocab_size, 200, 1)
        self.lr_sampler = lr.LearningRateTime(1.0)
        self.add_sampler_as_prior = False
        self.load_sampler = None
        self.write_sampler = None
        self.fix_sampler = False

        # learning rate
        self.lr_feat = lr.LearningRateEpochDelay(1.0)
        self.lr_net = lr.LearningRateEpochDelay(1.0)
        self.lr_logz = lr.LearningRateEpochDelay(1.0)
        self.opt_feat_method = 'sgd'
        self.opt_net_method = 'sgd'
        self.opt_logz_method = 'sgd'
        self.zeta_gap = 10
        self.max_epoch = 100

        # dbg
        self.write_dbg = False

    def get_initial_logz(self, c=None):
        if c is None:
            c = np.log(self.vocab_size)
        len_num = self.max_len - self.min_len + 1
        logz = c * (np.linspace(1, len_num, len_num))
        return logz

    def __str__(self):
        s = 'trf_IS'
        if self.prior_model_path is not None:
            s += '_priorlm'

        if self.add_sampler_as_prior:
            s += '_samplerlm'

        if self.feat_config is not None:
            s += '_' + str(self.feat_config)
        if self.net_config is not None:
            s += '_' + str(self.net_config)

        if self.norm_type != 'multiple':
            s += '_norm' + self.norm_type

        s += '_' + str(self.sampler_config)
        # s += '_globalNorm' if self.global_norm else '_sepNorm'
        # if not np.all(self.pi_0 == self.pi_true):
        #     s += '_pi0'
        return s


class TRF(object):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):
        self.config = config
        self.data = data
        self.logdir = logdir
        self.name = name

        # prior LM, q(x)
        self.priorlm = priorlm.LSTMLM(config.prior_model_path, device=device) \
            if config.prior_model_path is not None else priorlm.EmptyLM()

        # phi
        self.phi_feat = pot.FeatPhi(config.feat_config, data, config.opt_feat_method) \
            if config.feat_config is not None else pot.Base()
        self.phi_net = pot.NetPhi(config.net_config, data, config.opt_net_method, device=device) \
            if config.net_config is not None else pot.Base()

        # logZ
        if self.config.norm_type == 'multiple':
            self.norm_const = pot.Norm(config, data, config.opt_logz_method)
        elif self.config.norm_type == 'linear':
            self.norm_const = norm.NormLinear(config, data, config.opt_logz_method)
        else:
            raise TypeError('unknown norm_type=' + self.config.norm_type)

        # simulater
        if isinstance(config.sampler_config, sampler.LSTMLen.Config):
            self.sampler = sampler.LSTMLen(config, device)
        elif isinstance(config.sampler_config, sampler.LSTM.Config):
            self.sampler = sampler.LSTM(config, device)

        if self.config.add_sampler_as_prior:
            self.priorlm = self.sampler

        # learning rate
        self.cur_lr_feat = 1.0
        self.cur_lr_net = 1.0
        self.cur_lr_logz = 1.0
        self.cur_lr_sampler = 1.0

        # training info
        self.training_info = {'trained_step': 0,
                              'trained_epoch': 0,
                              'trained_time': 0}

        # debuger
        self.write_files = wb.FileBank(os.path.join(logdir, name + '.dbg'))
        # time recorder
        self.time_recoder = wb.clock()
        # default save name
        self.default_save_name = os.path.join(self.logdir, self.name + '.mod')

        # debug variables
        self.sample_cur_pi = np.zeros(self.config.max_len + 1)  # current pi
        self.sample_acc_pi = np.zeros(self.config.max_len + 1)  # accumulated count

    @property
    def global_step(self):
        return self.phi_net.global_step

    @property
    def session(self):
        return tf.get_default_session()

    def save(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[TRF] save to', fname)
        with open(fname + '.config', 'wt') as f:
            json.dump(self.training_info, f, indent=4)
            f.write('\n')
            self.config.save(f)
        self.phi_feat.save(fname)
        self.phi_net.save(fname)
        self.norm_const.save(fname + '.norm')

        if self.config.write_sampler is not None:
            self.sampler.save(self.config.write_sampler)

    def restore(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[TRF] restore from', fname)
        with open(fname + '.config', 'rt') as f:
            self.training_info = wb.json_load(f)
            print(json.dumps(self.training_info, indent=2))
        self.phi_feat.restore(fname)
        self.phi_net.restore(fname)
        self.norm_const.restore(fname + '.norm')

        if self.config.load_sampler is not None:
            self.sampler.restore(self.config.load_sampler)

    def restore_nce_model(self, fname):
        print('[TRF] restore nce model from', fname)
        self.phi_feat.restore(fname)
        self.phi_net.restore(fname)

        nce_norm = NormLinear(self.config, self.data)
        nce_norm.restore(fname + '.norm')
        logz = nce_norm.get_logz(np.linspace(self.config.min_len,
                                             self.config.max_len + 1,
                                             self.config.max_len - self.config.min_len + 1))
        self.norm_const.set_logz(logz)
        self.norm_const.set_logz1(self.true_logz(self.config.min_len)[0])

    def exist_model(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        return wb.exists(fname + '.norm')

    def phi(self, input_x, input_n):
        seq_list = reader.extract_data_from_array(input_x, input_n)
        return self.phi_feat.get_value(seq_list) + self.phi_net.get_value(seq_list)

    def normalize(self, phi, input_n, use_pi_true=True):
        if use_pi_true:
            logp_m = phi + np.log(self.config.pi_true[input_n]) - self.norm_const.get_logz(input_n)
        else:
            logp_m = phi + np.log(self.config.pi_0[input_n]) - self.norm_const.get_logz(input_n)

        return logp_m

    def logps(self, input_x, input_n, use_pi_true=True):

        phi = self.phi(input_x, input_n)
        seq_list = reader.extract_data_from_array(input_x, input_n)

        if np.any(input_n < self.config.min_len) or np.any(input_n > self.config.max_len):
            raise TypeError('min_len={}, max_len={} lens={}'.format(min(input_n), max(input_n), input_n))

        return self.normalize(phi + self.priorlm.get_log_probs(seq_list), input_n, use_pi_true)

    def get_log_probs(self, seq_list, is_norm=True, use_pi_true=True, minibatch=100):
        seqs, indexs = self.data.cut_data_to_length(seq_list,
                                                    maxlen=self.config.max_len)

        logps = np.zeros(len(seqs))

        for i in range(0, len(seqs), minibatch):
            input_x, input_n = reader.produce_data_to_array(seqs[i: i + minibatch])
            if is_norm:
                logps[i: i + minibatch] = self.logps(input_x, input_n, use_pi_true)
            else:
                logps[i: i + minibatch] = self.phi(input_x, input_n)

        res = []
        for idx_b, idx_e in indexs:
            res.append(np.sum(logps[idx_b: idx_e]))

        return np.array(res)

    def rescore(self, seq_list):
        return -self.get_log_probs(seq_list)

    def eval(self, seq_list):
        logps = self.get_log_probs(seq_list)
        nll = -np.mean(logps)
        words = np.sum([len(x) - 1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)

        return nll, ppl

    def true_logz(self, max_len=None):
        if max_len is None:
            max_len = self.config.max_len

        logz = np.zeros(max_len - self.config.min_len + 1)
        for l in range(self.config.min_len, max_len + 1):
            x_batch = [x for x in sp.SeqIter(l, self.config.vocab_size,
                                             beg_token=self.config.beg_token,
                                             end_token=self.config.end_token)]
            logz[l - self.config.min_len] = logsumexp(self.get_log_probs(x_batch, is_norm=False))
        return logz

    def draw(self, n):
        sample_list = self.sampler.generate(n)

        with self.time_recoder.recode('write_sample'):
            f = self.write_files.get('sample')
            for x in sample_list:
                log.write_seq(f, x)
            f.flush()

        return sample_list

    def get_sample_scale(self, sample_list):
        n = len(sample_list)

        lengths = [len(x) for x in sample_list]
        logr = np.log(self.config.pi_true[lengths]) - np.log(self.config.pi_0[lengths])
        logp = self.get_log_probs(sample_list, is_norm=True, use_pi_true=False)
        logq = self.sampler.get_log_probs(sample_list)

        log_weight = logp - logq - np.log(n)
        log_weight -= logsumexp(log_weight)

        scale_for_param = np.exp(log_weight + logr)
        scale_for_zeta = np.exp(log_weight)

        scale_for_aux = scale_for_zeta

        # output the scalar
        f = self.write_files.get('sample_scale')
        f.write('step={}\n'.format(self.training_info['trained_step']))
        f.write('[len] [scale] [scale2] [scale3] [logp  ] [logql  ] [seq  ]\n')
        for i in range(len(sample_list)):
            f.write('{:<5} {:<7.5f} {:<8.5f} {:<8.5f} {:<8.2f} {:<8.2f} '.format(
                len(sample_list[i]), scale_for_param[i], scale_for_zeta[i], scale_for_aux[i],
                logp[i], logq[i]))
            f.write('[' + ' '.join(str(w) for w in sample_list[i]) + ']\n')
        f.flush()

        return scale_for_param, \
               scale_for_zeta, \
               scale_for_aux, \
               np.mean(logp - logq) - logsumexp(logp - logq)

    def get_sample_scale_sep(self, sample_list):
        n = len(sample_list)

        lengths = [len(x) for x in sample_list]
        phi = self.get_log_probs(sample_list, is_norm=False)
        logp = self.normalize(phi, lengths, use_pi_true=False)
        logpl = phi - self.norm_const.get_logz(lengths)
        logql, logq = self.sampler.get_log_probs_duel(sample_list)
        logw = phi - logql

        logw_dict = dict()
        for j, w in zip(lengths, logw):
            a = logw_dict.setdefault(j, [])
            a.append(w)

        logzl = np.zeros(self.config.max_len+1)
        nl = np.zeros(self.config.max_len+1)
        for j, a in logw_dict.items():
            logzl[j] = logsumexp(a) - np.log(len(a))  # 1/n_l * sum_x^l e^phi(x^l) / q_l(x^l)
            nl[j] = len(a)

        log_scale_for_param = logw - logzl[lengths] - np.log(n) + \
                              np.log(self.config.pi_true[lengths]) - np.log(self.sampler.len_prob[lengths])
        log_scale_for_sampler = logw - logzl[lengths] - np.log(n)

        # log_scale_for_zeta = logp - logq
        # log_scale_for_zeta -= logsumexp(log_scale_for_zeta)
        log_scale_for_zeta = logp - logql - np.log(nl[lengths])
        log_scale_for_zeta -= logsumexp(log_scale_for_zeta)

        # # update logzl
        # old_logzl = self.norm_const.get_logz()
        # new_logzl = old_logzl + self.cur_lr_logz * (logzl[self.config.min_len:] - old_logzl)
        # self.norm_const.set_logz(new_logzl)

        # output the scalar
        f = self.write_files.get('sample_scale')
        f.write('step={}\n'.format(self.training_info['trained_step']))
        f.write('[len] [scale] [scale2] [logp  ] [logql  ] [phi   ] [seq  ]\n')
        for i in range(len(sample_list)):
            f.write('{:<5} {:<7.5f} {:<8.5f} {:<8.2f} {:<8.2f} {:<8.2f} '.format(
                len(sample_list[i]), np.exp(log_scale_for_param[i]), np.exp(log_scale_for_zeta[i]),
                logp[i], logql[i], phi[i]))
            f.write('[' + ' '.join(str(w) for w in sample_list[i]) + ']\n')
        f.flush()

        return np.exp(log_scale_for_param), \
               np.exp(log_scale_for_zeta), \
               np.exp(log_scale_for_sampler), \
               np.mean(logpl-logql), \
               logzl

    def get_train_sacle(self, data_list):
        return np.ones(len(data_list)) / len(data_list)

    def update(self, data_list, sample_list):
        info = OrderedDict()

        # training scales
        data_scalar = self.get_train_sacle(data_list)

        sample_inputs, sample_lengths = reader.produce_data_to_array(sample_list)
        logr = np.log(self.config.pi_true[sample_lengths]) - np.log(self.config.pi_0[sample_lengths])
        model_phi = self.phi(sample_inputs, sample_lengths)
        model_prior = self.priorlm.get_log_probs(sample_list)
        model_logps = self.normalize(model_phi + model_prior, sample_lengths, use_pi_true=False)
        noise_logps = self.sampler.get_log_probs(sample_list)
        log_weight = model_logps - noise_logps - np.log(len(sample_lengths))
        log_weight -= logsumexp(log_weight)

        sample_scale_for_param = np.exp(log_weight + logr)
        sample_scale_for_zeta = np.exp(log_weight)

        # # sample scales
        # if self.config.global_norm:
        #     sample_scale, scale_zeta, scale_aux, kl = self.get_sample_scale(sample_list)
        # else:
        #     sample_scale, scale_zeta, scale_aux, kl, logz = self.get_sample_scale_sep(sample_list)
        #
        info['ESR'] = np.sum(sample_scale_for_param >= 1e-3) / len(sample_scale_for_param)
        info['ESR2'] = np.sum(sample_scale_for_zeta >= 1e-3) / len(sample_scale_for_zeta)
        # # info['sample_scalar_sum'] = np.sum(sample_scale * np.array([len(x) for x in sample_list]))
        # # info['data_scalar_sum'] = np.sum(data_scalar * np.array([len(x) for x in data_list]))
        # info['kl'] = kl

        # update feat-phi
        with self.time_recoder.recode('update_feat'):
            info['feat_g_norm'] = self.phi_feat.update(data_list, data_scalar,
                                                       sample_list, sample_scale_for_param,
                                                       learning_rate=self.cur_lr_feat)

        # update net-phi
        with self.time_recoder.recode('update_net'):
            info['net_g_norm'] = self.phi_net.update(data_list, data_scalar,
                                                     sample_list, sample_scale_for_param,
                                                     learning_rate=self.cur_lr_net)

        # update zeta
        with self.time_recoder.recode('update_zeta'):
            info['zeta_g_norm'] = self.norm_const.update(sample_list, sample_scale_for_zeta,
                                                         learning_rate=self.cur_lr_logz)
            true_logz1 = self.true_logz(self.config.min_len)[0]
            self.norm_const.set_logz1(true_logz1)

        # update sampling lstm
        with self.time_recoder.recode('update_sampler'):
            if not self.config.fix_sampler:
                self.sampler.update(data_list, np.ones(len(data_list))/100, lr=self.cur_lr_sampler, batch_size=100)

        # update dbg info
        self.sample_cur_pi.fill(0)
        true_sample_pi = np.zeros_like(self.sample_cur_pi)
        for x, s in zip(sample_list, sample_scale_for_zeta):
            self.sample_cur_pi[len(x)] += s
            true_sample_pi[len(x)] += 1.0 / len(sample_list)
        self.sample_acc_pi += self.sample_cur_pi * len(sample_list)

        acc_pi = self.sample_acc_pi / np.sum(self.sample_acc_pi)
        info['pi_dist'] = np.arccos(np.dot(acc_pi, self.config.pi_0) /
                                    np.linalg.norm(acc_pi) / np.linalg.norm(self.config.pi_0))

        #  write zeta, logz, pi
        f = self.write_files.get('zeta')
        f.write('step={}\n'.format(self.training_info['trained_step']))
        log.write_array(f, self.config.pi_0[self.config.min_len:], name='pi_0  ')
        log.write_array(f, acc_pi[self.config.min_len:], name='acc_pi')
        log.write_array(f, self.sample_cur_pi[self.config.min_len:], name='cur_pi')
        log.write_array(f, self.sampler.len_prob[self.config.min_len:], name='smp_pi')
        log.write_array(f, self.norm_const.zeta, name='zeta  ')
        log.write_array(f, self.norm_const.get_gradient(sample_list, sample_scale_for_zeta), name='grad  ')
        log.write_array(f, self.norm_const.get_logz(), name='logz  ')
        # log.write_array(f, self.norm_const.a, name='a     ')
        f.flush()

        if self.config.write_dbg:
            f = self.write_files.get('noise')
            f.write('step={}\n'.format(self.training_info['trained_step']))
            f.write('[model_logp] [noise_logp] [model_phi] [model_prior] [ scale ] [ seq ]\n')
            for i, s in enumerate(sample_list):
                f.write('{:<12.5f} {:<12.5f} {:<12.5f} {:<12.5f} {:<12.5f} '.format(
                    model_logps[i], noise_logps[i], model_phi[i], model_prior[i], sample_scale_for_param[i]))
                f.write('[' + ' '.join(str(w) for w in s) + ']\n')
            f.flush()

        return info

    def initialize(self):
        # print the txt information
        for d, name in zip(self.data.datas, ['train', 'valid', 'test']):
            info = wb.TxtInfo(d)
            print('[TRF]', name, ':', str(info))

        # load prior
        self.priorlm.initialize()

        # create features
        self.phi_feat.initialize()
        self.phi_net.initialize()

        # print parameters
        print('[TRF] feat_num = {:,}'.format(self.phi_feat.get_param_num()))
        print('[TRF] net_num  = {:,}'.format(self.phi_net.get_param_num()))

        self.sampler.initialize()

        if self.config.load_sampler is not None:
            print('[TRF] load sampler from {}'.format(self.config.load_sampler))
            self.sampler.restore(self.config.load_sampler)

    def train(self, print_per_epoch=0.1, operation=None):

        # initialize
        self.initialize()

        if self.exist_model():
            self.restore()

        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]
        test_list = self.data.datas[2]

        print('[TRF] [Train]...')
        epoch_contain_step = len(train_list) // self.config.train_batch_size

        time_beginning = time.time()
        model_train_nll = []
        model_train_nll_step = 10

        step = self.training_info['trained_step']
        epoch = step / epoch_contain_step
        print_next_epoch = int(epoch)

        while epoch < self.config.max_epoch:

            ###########################
            # extra operations
            ###########################
            with self.time_recoder.recode('operation'):
                if operation is not None:
                    operation.run(step, epoch)

            # update training information
            self.training_info['trained_step'] = step
            self.training_info['trained_epoch'] = epoch
            self.training_info['trained_time'] = (time.time() - time_beginning) / 60

            if step % epoch_contain_step == 0:
                np.random.shuffle(train_list)
                self.save()

            # get empirical list
            data_beg = step % epoch_contain_step * self.config.train_batch_size
            data_list = train_list[data_beg: data_beg + self.config.train_batch_size]

            # draw samples
            with self.time_recoder.recode('sample'):
                sample_list = self.draw(self.config.sample_batch_size)

            # update paramters
            with self.time_recoder.recode('update'):
                # learining rate
                self.cur_lr_feat = self.config.lr_feat.get_lr(step + 1, epoch)
                self.cur_lr_net = self.config.lr_net.get_lr(step + 1, epoch)
                self.cur_lr_logz = self.config.lr_logz.get_lr(step + 1, epoch)
                self.cur_lr_sampler = self.config.lr_sampler.get_lr(step + 1, epoch)
                # update
                update_info = self.update(data_list, sample_list)

            ##########################
            # update step
            ##########################
            step += 1
            epoch = step / epoch_contain_step

            # evaulate the nll and KL-distance
            with self.time_recoder.recode('eval_train_nll'):
                model_train_nll.append(self.eval(data_list)[0])

            # print
            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                time_since_beg = (time.time() - time_beginning) / 60

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]
                    # model_test_nll = self.eval(test_list)[0]
                    simul_valid_nll = self.sampler.eval_nll(valid_list)

                # cmp the normalization constants
                # true_logz1 = self.true_logz(self.config.min_len)[0]
                # self.norm_const.set_logz1(true_logz1)

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                info['lr_feat'] = self.cur_lr_feat
                info['lr_net'] = self.cur_lr_net
                info['lr_logz'] = self.cur_lr_logz
                info['lr_sampler'] = self.cur_lr_sampler
                info['logz1'] = self.norm_const.logz1
                info['train'] = np.mean(model_train_nll[-epoch_contain_step:])
                info['valid'] = model_valid_nll
                # info['test'] = model_test_nll
                info['sampler_valid'] = simul_valid_nll
                info.update(update_info)

                log.print_line(info)

                print('[end]')

                # write time
                f = self.write_files.get('time')
                f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, time_since_beg))
                f.write(
                    ' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.time_recoder.items()]) + '\n')
                f.flush()
