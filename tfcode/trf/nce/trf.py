import tensorflow as tf
import time
import json
import os
from collections import OrderedDict

from base import *
from lm import *
from trf.common import priorlm, feat
from trf.nce import noise, pot


class Config(wb.Config):
    def __init__(self, data):
        Config.value_encoding_map[lr.LearningRate] = str

        self.min_len = data.get_min_len()
        self.max_len = data.get_max_len()
        self.vocab_size = data.get_vocab_size()
        self.pi_true = data.get_pi_true()
        self.pi_0 = data.get_pi0()
        self.beg_token = data.get_beg_token()
        self.end_token = data.get_end_token()

        self.word_average = True

        # prior model path
        self.prior_model_path = None

        # interpolate model
        self.interpolate_model = None
        self.interpolate_factor_with_noise = None

        # for discrete features
        self.feat_config = pot.FeatConfig(data)

        # for network features
        self.net_config = pot.NetConfig(data)

        # init zeta
        self.init_logz = [0, np.log(self.vocab_size)]

        # nce
        self.batch_size = 10
        self.noise_factor = 10
        self.noise_sampler = '2gram'

        # learning rate
        self.lr_feat = lr.LearningRateEpochDelay(1.0)
        self.lr_net = lr.LearningRateEpochDelay(1.0)
        self.lr_logz = lr.LearningRateEpochDelay(1.0)
        self.opt_feat_method = 'sgd'
        self.opt_net_method = 'sgd'
        self.opt_logz_method = 'sgd'
        self.var_gap = 1e-3  # if > 0 , then compute the variance to rescale the gradient of logz
        self.max_epoch = 100

        self.data_sampler = None

        # dbg
        self.write_dbg = False

    def get_initial_logz(self, c=None):
        if c is None:
            c = np.log(self.vocab_size)
        len_num = self.max_len - self.min_len + 1
        logz = c * (np.linspace(1, len_num, len_num))
        return logz

    def __str__(self):
        s = 'trf_nce{}'.format(self.noise_factor)
        if self.prior_model_path is not None:
            s += '_priorlm'

        if self.feat_config is not None:
            s += '_' + str(self.feat_config)
        if self.net_config is not None:
            s += '_' + str(self.net_config)

        s += '_noise{}'.format(self.noise_sampler.split(':')[0])

        if self.data_sampler is not None:
            s += '_data{}'.format(self.data_sampler.split(':')[0])
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
        # self.norm_const = pot.Norm(config, data, config.opt_logz_method)
        self.norm_const = pot.NormLinear(config, data, config.opt_logz_method)

        # noise sampler
        self.noise_sampler = noise.create_noise_sampler(config.noise_sampler, noise.Config(config),
                                                        data,
                                                        device=device,
                                                        logdir=wb.mkdir(os.path.join(logdir, 'sampler')))

        # data sampler
        self.data_sampler = None
        if self.config.data_sampler is not None:
            if self.config.data_sampler == self.config.noise_sampler:
                self.data_sampler = self.noise_sampler
            else:
                sampler_config = noise.Config(config)
                sampler_config.pack_size = self.config.batch_size
                self.data_sampler = noise.create_noise_sampler(config.data_sampler, sampler_config, data,
                                                               device=device,
                                                               logdir=wb.mkdir(os.path.join(logdir, 'sampler')))

        # learning rate
        self.cur_lr_feat = 1.0
        self.cur_lr_net = 1.0
        self.cur_lr_logz = 1.0

        # training info
        self.training_info = {'trained_step': 0,
                              'trained_epoch': 0,
                              'trained_time': 0}

        # debuger
        self.write_files = wb.FileBank(os.path.join(logdir, name + '.dbg'))
        # time recorder
        self.time_recoder = wb.clock()
        self.default_save_name = os.path.join(self.logdir, self.name + '.mod')

        self.unigram = data.get_unigram()

    @property
    def global_step(self):
        return self.phi_net.global_step

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

    def exist_model(self, fname=None):
        if fname is None:
            fname = self.default_save_name

        return wb.exists(fname + '.norm')

    def phi(self, seq_list, for_eval=True):
        if for_eval:
            return self.phi_feat.get_value(seq_list) + self.phi_net.get_value(seq_list)
        else:
            return self.phi_feat.get_value_for_train(seq_list) + self.phi_net.get_value_for_train(seq_list)

    def logps(self, seq_list, for_eval=True):
        phi = self.phi(seq_list, for_eval)
        lengths = np.array([len(x) for x in seq_list])

        if np.any(lengths < self.config.min_len) or np.any(lengths > self.config.max_len):
            raise TypeError('min_len={}, max_len={} lens={}'.format(min(lengths), max(lengths), lengths))

        logp_m = phi + self.priorlm.get_log_probs(seq_list) + \
                 np.log(self.config.pi_true[lengths]) - self.norm_const.get_logz(lengths)

        if self.config.interpolate_factor_with_noise is not None:
            logp_n = self.noise_sampler.noise_logps(seq_list)
            logp = np.logaddexp(logp_n + np.log(self.config.interpolate_factor_with_noise),
                                logp_m + np.log(1-self.config.interpolate_factor_with_noise))

            return logp
        else:
            return logp_m

    def get_log_probs(self, seq_list, is_norm=True):

        # pad </s> to the head of each sentences
        logps = np.zeros(len(seq_list))
        minibatch = self.config.batch_size * self.config.noise_factor
        for i in range(0, len(seq_list), minibatch):

            seqs = [x for x in seq_list[i: i+minibatch]]

            if is_norm:
                logps[i: i + minibatch] = self.logps(seqs)
            else:
                logps[i: i + minibatch] = self.phi(seqs)

        return logps

        # seqs, indexs = self.data.cut_data_to_length(seq_list,
        #                                             maxlen=self.config.max_len)
        #
        # logps = np.zeros(len(seqs))
        #
        # minibatch = self.config.batch_size * self.config.noise_factor
        # for i in range(0, len(seqs), minibatch):
        #     if is_norm:
        #         logps[i: i+minibatch] = self.logps(seqs[i: i+minibatch], for_eval)
        #     else:
        #         logps[i: i+minibatch] = self.phi(seqs[i: i+minibatch], for_eval)
        #
        # res = []
        # for idx_b, idx_e in indexs:
        #     res.append(np.sum(logps[idx_b: idx_e]))
        #
        # return np.array(res)

    def eval(self, seq_list):
        logps = self.get_log_probs(seq_list)
        nll = -np.mean(logps)
        words = np.sum([len(x)-1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)

        return nll, ppl

    def true_logz(self, max_len=None):
        if max_len is None:
            max_len = self.config.max_len

        logz = np.zeros(max_len - self.config.min_len + 1)
        for l in range(self.config.min_len, max_len+1):
            x_batch = [x for x in sp.SeqIter(l, self.config.vocab_size,
                                             beg_token=self.config.beg_token)]
            logz[l-self.config.min_len] = sp.log_sum(self.get_log_probs(x_batch, False))
        return logz

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

        if self.config.interpolate_factor_with_noise is not None:
            w *= (1 - self.config.interpolate_factor_with_noise * np.exp(logpn - logpm))

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

    def update_lr(self, step, epoch):
        # learining rate
        self.cur_lr_feat = self.config.lr_feat.get_lr(step+1, epoch)
        self.cur_lr_net = self.config.lr_net.get_lr(step+1, epoch)
        self.cur_lr_logz = self.config.lr_logz.get_lr(step+1, epoch)

    def update_phi(self, seq_list, cluster_weights, cluster_m):
        # update parameters
        with self.time_recoder.recode('update_feat'):
            self.phi_feat.update(seq_list, cluster_weights, cluster_m, learning_rate=self.cur_lr_feat)

        with self.time_recoder.recode('update_net'):
            self.phi_net.update(seq_list, cluster_weights, cluster_m, learning_rate=self.cur_lr_net)

    def update(self, data_list):

        # generate noise samples
        with self.time_recoder.recode('sampling'):

            if self.data_sampler is not None:
                n = len(data_list)
                data_list_batch, _ = self.data_sampler.get()
                data_list = data_list[0: n//2] + data_list_batch[0: n - n//2]
            data_logpn = self.noise_sampler.noise_logps(data_list)

            sample_list, sample_logpn = self.noise_sampler.get(data_list)
            assert len(sample_list) == self.config.batch_size * self.config.noise_factor

            seq_list = data_list + sample_list
            noise_logps = np.concatenate([data_logpn, sample_logpn])
            data_num = len(data_list)
            seq_lens = [len(x) for x in seq_list]

        with self.time_recoder.recode('loss'):
            model_logps = self.logps(seq_list, for_eval=False)  # for training to calculate the logp
            cluster_weights = self.cmp_cluster_weight(model_logps, noise_logps, data_num, seq_lens)
            # cluster_m = self.cmp_cluster_m(model_logps, noise_logps, data_num, seq_lens)

            loss_all = self.cmp_cluster_loss(model_logps, noise_logps, data_num, seq_lens)
            loss = np.sum(loss_all)

            # cluster_weights, loss = self.get_cluster_weight_and_loss(seq_list, data_num, noise_logps)

        # update phi
        self.update_phi(seq_list, cluster_weights, cluster_m=None)

        # update zeta
        self.norm_const.update(seq_list, cluster_weights, cluster_m=None, learning_rate=self.cur_lr_logz)

        if self.config.write_dbg:
            f = self.write_files.get('dbg')
            log.write_array(f, cluster_weights, 'w    ')
            # log.write_array(f, cluster_m, 'm    ')
            log.write_array(f, seq_lens, 'len  ')
            log.write_array(f, model_logps, 'logpm')
            log.write_array(f, noise_logps, 'logpn')
            log.write_array(f, self.cmp_cluster_logps(model_logps, noise_logps), 'logpc')
            log.write_array(f, loss_all, 'loss ')
            f.write('\n')
            f.flush()

            f = self.write_files.get('noise')
            for s in sample_list:
                log.write_seq(f, s)

        f = self.write_files.get('logz')
        logz = self.norm_const.get_var()
        grad = self.norm_const.get_gradient(seq_list, cluster_weights)
        # vars = self.norm_const.get_variance(seq_list, cluster_m)
        log.write_array(f, logz, 'logz')
        log.write_array(f, grad, 'grad')
        # log.write_array(f, vars, 'vars')
        f.write('\n')
        f.flush()

        print_infos = {'sumw': np.sum(cluster_weights)}
        return loss, print_infos

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

        # start sampler
        self.noise_sampler.start()
        if self.data_sampler is not None and self.data_sampler != self.noise_sampler:
            self.data_sampler.start()

    def extand_data_seqs(self, data_seqs):
        extra_data_seqs = []

        for seq in data_seqs:
            a = np.arange(1, len(seq)-1)
            np.random.shuffle(a)
            a = a[0: len(a)//2]
            new_seq = np.array(seq)
            if len(a) > 0:
                new_seq[a] = np.random.choice(self.config.vocab_size, size=len(a), p=self.unigram)
            extra_data_seqs.append(new_seq.tolist())

        return data_seqs + extra_data_seqs

    def train(self, print_per_epoch=0.1, operation=None):

        # initialize
        self.initialize()

        if self.exist_model():
            self.restore()

        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]
        test_list = self.data.datas[2]

        print('[TRF] [Train]...')
        epoch_contain_step = int(len(train_list) / self.config.batch_size)

        time_beginning = time.time()
        model_train_nll = []
        model_train_loss = []

        step = self.training_info['trained_step']
        epoch = step / epoch_contain_step
        print_next_epoch = int(epoch)
        while epoch < self.config.max_epoch:

            # update training information
            self.training_info['trained_step'] = step
            self.training_info['trained_epoch'] = epoch
            self.training_info['trained_time'] = (time.time() - time_beginning) / 60

            # shuffle the data
            if step % epoch_contain_step == 0:
                np.random.shuffle(train_list)
                self.save()

            # current data sequences
            data_seqs = train_list[
                        step % epoch_contain_step * self.config.batch_size:
                        (step % epoch_contain_step + 1) * self.config.batch_size
                        ]
            # data_seqs = self.extand_data_seqs(data_seqs)

            # update parameters
            with self.time_recoder.recode('update'):
                # update lr
                self.update_lr(step, epoch)
                # update parameters
                loss, print_infos = self.update(data_seqs)

            # compute the nll on training set
            with self.time_recoder.recode('train_eval'):
                model_train_nll.append(self.eval(data_seqs)[0])
                model_train_loss.append(loss)

            # update steps
            step += 1
            epoch = step / epoch_contain_step

            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]
                    model_test_nll = self.eval(test_list)[0]

                time_since_beg = (time.time() - time_beginning) / 60

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                info['lr_feat'] = '{:.2e}'.format(self.cur_lr_feat)
                info['lr_net'] = '{:.2e}'.format(self.cur_lr_net)
                info['lr_logz'] = '{:.2e}'.format(self.cur_lr_logz)
                # info['logz1'] = self.true_logz(self.config.min_len)[0]
                info['loss'] = np.mean(model_train_loss[-epoch_contain_step:])
                info.update(print_infos)
                info['train'] = np.mean(model_train_nll[-epoch_contain_step:])
                info['valid'] = model_valid_nll
                info['test'] = model_test_nll
                log.print_line(info)

                print('[end]')

                #####################################
                # write time
                #####################################
                # write to file
                f = self.write_files.get('time')
                f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, time_since_beg))
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.time_recoder.items()]) + '\n')
                f.flush()

            ###########################
            # extra operations
            ###########################
            if operation is not None:
                operation.run(step, epoch)

        # stop the sub-process
        self.noise_sampler.release()


class DefaultOps(wb.Operation):
    def __init__(self, m, nbest_or_nbest_file_tuple, scale_vec=np.linspace(0.1, 1.0, 10)):
        self.m = m
        self.scale_vec = scale_vec

        if isinstance(nbest_or_nbest_file_tuple, tuple):
            print('[%s.%s] input the nbest files.' % (__name__, self.__class__.__name__))
            self.nbest_cmp = reader.NBest(*nbest_or_nbest_file_tuple)
        else:
            print('[%s.%s] input nbest computer.' % (__name__, self.__class__.__name__))
            self.nbest_cmp = nbest_or_nbest_file_tuple

        self.wer_next_epoch = 0
        self.wer_per_epoch = 1.0
        self.write_models = wb.mkdir(os.path.join(self.m.logdir, 'wer_results'))

    def run(self, step, epoch):
        super().run(step, epoch)

        if epoch >= self.wer_next_epoch:
            self.wer_next_epoch = int(epoch + self.wer_per_epoch)

            # resocring
            time_beg = time.time()
            self.nbest_cmp.lmscore = -self.m.get_log_probs(self.nbest_cmp.get_nbest_list(self.m.data))
            rescore_time = time.time() - time_beg

            # compute wer
            time_beg = time.time()
            wer = self.nbest_cmp.wer(lmscale=self.scale_vec)
            wer_time = time.time() - time_beg

            wb.WriteScore(self.write_models + '/epoch%.2f' % epoch + '.lmscore', self.nbest_cmp.lmscore)
            print('epoch={:.2f} test_wer={:.2f} lmscale={} '
                  'rescore_time={:.2f}, wer_time={:.2f}'.format(
                   epoch, wer, self.nbest_cmp.lmscale,
                   rescore_time / 60, wer_time / 60))

            res = wb.FRes(os.path.join(self.m.logdir, 'wer_per_epoch.log'))
            res_name = 'epoch%.2f' % epoch
            res.Add(res_name, ['lm-scale'], [self.nbest_cmp.lmscale])
            res.Add(res_name, ['wer'], [wer])


class TRF2(TRF):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf',
                 pretrain_lstm_path=None):
        super().__init__(config, data, logdir, device, name)

        assert self.phi_net is not None
        assert self.phi_net.config.structure_type == 'rnn'

        self.pretrain_lstm_path = pretrain_lstm_path

        # create saver
        var_dict = {}
        for v in self.phi_net.train_net.vars:
            name = v.name.split(':')[0]
            tag = name.split('/')[0]
            name = name.replace(tag, 'lstmlm')
            name = name.replace('final_pred', 'BNCELoss')
            var_dict[name] = v
        self.pretrain_saver = tf.train.Saver(var_dict)

    def initialize(self):
        super().initialize()
        if self.pretrain_lstm_path is not None:
            self.load_pretrain_vairables(tf.get_default_session(), self.pretrain_lstm_path)

    def load_pretrain_vairables(self, session, fname):
        # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        # # List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
        # print_tensors_in_checkpoint_file(file_name=fname, tensor_name='', all_tensors=False)
        print('[TRF] load pretrain variables in ', fname)
        self.pretrain_saver.restore(session, fname)
