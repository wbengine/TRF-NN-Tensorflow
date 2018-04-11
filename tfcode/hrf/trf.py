import tensorflow as tf
import os
import json
import time
from copy import deepcopy
from collections import OrderedDict
from base import *
from lm import *

from trf.common import feat2 as feat
from . import crf, mcmc, pot, norm


class Config(crf.Config):
    def __init__(self, data, pretrain_word=False):
        super().__init__(data)

        Config.value_encoding_map[lr.LearningRate] = str

        # pretrain
        self.pretrain_word = pretrain_word

        self.min_len = data.get_min_len()
        self.max_len = data.get_max_len()
        self.pi_true = data.get_pi_true()
        self.pi_0 = data.get_pi0(self.pi_true)
        self.word_vocab_size = data.get_vocab_size()
        self.tag_vocab_size = data.get_tag_size()
        self.beg_tokens = data.get_beg_tokens()  # [word_beg_token, tag_beg_token]
        self.end_tokens = data.get_end_tokens()  # [word_end_token, tag_end_token]

        # prior model path
        self.prior_model_path = None

        # for words
        self.word_config = None  # features

        # init zeta
        self.init_logz = self.get_initial_logz()

        # init CRF
        self.load_crf_model = None
        self.fix_crf_model = False

        self.load_trf_model = None
        self.fix_trf_model = False

        # AugSA
        self.train_batch_size = 1000
        self.sample_batch_size = 100
        self.chain_num = 100
        self.multiple_trial = 1
        self.sample_sub = 1
        self.jump_width = 3
        self.auxiliary_type = 'lstm'
        self.auxiliary_config = lstmlm.Config(self.word_vocab_size, 200, 1)

        # learning rate
        self.lr_word = lr.LearningRateEpochDelay(1.0)
        self.lr_logz = lr.LearningRateEpochDelay(1.0)
        self.opt_word = 'sgd'
        self.opt_logz = 'sgd'
        self.zeta_gap = 10
        self.max_epoch = 100

        # dbg
        self.write_dbg = False

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
            if not self.pretrain_word:
                c = np.log(self.word_vocab_size) + np.log(self.tag_vocab_size)
            else:
                c = np.log(self.word_vocab_size)
        len_num = self.max_len - self.min_len + 1
        logz = c * (np.linspace(1, len_num, len_num))
        return logz

    def __str__(self):
        s = 'hrf_sa{}'.format(self.chain_num)
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

        if self.pretrain_word:
            s = s.replace('hrf_', 'trf_')

        return s


class TRF_word(object):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):

        self.config = config
        self.name = name
        self.logdir = logdir
        self.data = data

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

        # phi for words
        self.phi_word = pot.create(config.word_config, data.datas[0], config.opt_word, device)

        # logZ
        self.norm_const = norm.Norm(config, data, config.opt_logz)

        # length jump probabilities
        self.gamma = sp.len_jump_distribution(self.config.min_len,
                                              self.config.max_len,
                                              self.config.jump_width)

        self.mcmc = mcmc.RJMCMC(q_config=self.config.auxiliary_config,
                                gamma=self.gamma,
                                multiple_trial=self.config.multiple_trial,
                                sample_sub=self.config.sample_sub,
                                end_tokens=self.config.end_tokens,
                                fun_logps=lambda x: self.logps(x, for_eval=False),
                                fun_propose_tag=None,
                                write_files=self.write_files,
                                device=device)

        # sample sequences
        self.mcmc_seqs = [seq.Seq(level=2).random(
                                min_len=self.config.min_len,
                                max_len=self.config.max_len,
                                vocab_sizes=[self.config.word_vocab_size, self.config.tag_vocab_size],
                                beg_tokens=self.config.beg_tokens,
                                end_tokens=self.config.end_tokens,
                                pi=self.config.pi_true)
                            for _ in range(self.config.chain_num)]
        self.mcmc_states = None

        # learning rate
        self.cur_lr_word = 1.0
        self.cur_lr_logz = 1.0

        # debug variables
        self.sample_cur_pi = np.zeros(self.config.max_len + 1)  # current pi
        self.sample_acc_count = np.zeros(self.config.max_len + 1)  # accumulated count

    def save(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        self.phi_word.save(fname)
        self.norm_const.save(fname + '.norm')

    def restore(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        self.phi_word.restore(fname)
        self.norm_const.restore(fname + '.norm')

    @property
    def session(self):
        return tf.get_default_session()

    def phi(self, seq_list):
        return self.phi_word.get_value(seq_list)

    def normalize(self, logp, lengths, for_eval=True):
        if for_eval:
            # using pi_true
            logp_m = logp + np.log(self.config.pi_true[lengths]) - self.norm_const.get_logz(lengths)
        else:
            # using pi_0
            logp_m = logp + np.log(self.config.pi_0[lengths]) - self.norm_const.get_logz(lengths)

        return logp_m

    def logps(self, seq_list, for_eval=True):
        phi = self.phi(seq_list)
        seq_lens = np.array([len(s) for s in seq_list])
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
        return -self.get_logpxs(x_list)

    def get_logpxs(self, x_list, is_norm=True, for_eval=True, batch_size=100):
        seq_list = [seq.Seq(x) for x in x_list]
        return self.get_log_probs(seq_list, is_norm, for_eval, batch_size)

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

    def initialize(self):
        print('[TRF] init word features.')
        self.phi_word.initialize()

        # print parameters
        print('[TRF] word_feat_num = {:,}'.format(self.phi_word.get_param_num()))

    def sample(self, seq_list, states_list):

        with self.time_recoder.recode('local_jump'):
            seq_list, states_list = self.mcmc.local_jump(seq_list, states_list)

        with self.time_recoder.recode('markov_move'):
            seq_list, states_list = self.mcmc.markov_move(seq_list, states_list)

        # monitor the hidden states

        # true_states = self.mcmc.update_hidden_state(seq_list)
        # d = self.mcmc.hidden_states_diff(true_states, states_list)
        # f = self.write_files.get('mcmc.dbg')
        # f.write('len={} d={}\n'.format([len(s) for s in seq_list], d))
        # f.flush()

        return seq_list, states_list

    def draw(self, n):

        self.mcmc.reset_dbg()

        # init the sample states
        # as at each iteration, the variables of lstm will be updated
        # the hidden states need to be uptated too
        self.mcmc_states = self.mcmc.update_hidden_state(self.mcmc_seqs)

        # draw samples
        res_seq_list = []
        for _ in range(n // len(self.mcmc_seqs)):
            self.mcmc_seqs, self.mcmc_states = self.sample(self.mcmc_seqs, self.mcmc_states)
            res_seq_list += seq.list_copy(self.mcmc_seqs)

        self.mcmc.update_dbg()

        with self.time_recoder.recode('write_sample'):
            f = self.write_files.get('sample')
            for s in res_seq_list:
                f.write(str(s))

        return res_seq_list

    def update(self, data_list, sample_list):
        # compute the scalars
        data_scalar = np.ones(len(data_list)) / len(data_list)
        sample_len = np.array([len(x) for x in sample_list])
        sample_facter = np.array(self.config.pi_true[self.config.min_len:]) / \
                        np.array(self.config.pi_0[self.config.min_len:])
        sample_scalar = sample_facter[sample_len - self.config.min_len] / len(sample_list)

        # update phi

        with self.time_recoder.recode('update_' + phi_name):
            self.phi_word.update(data_list, data_scalar, sample_list, sample_scalar, learning_rate=self.cur_lr_word)

        # update zeta
        with self.time_recoder.recode('update_logz'):
            self.norm_const.update(sample_list, learning_rate=self.cur_lr_logz)
            self.norm_const.set_logz1(self.get_true_logz(self.config.min_len)[0])

        # update simulater
        with self.time_recoder.recode('update_simulater'):
            self.mcmc.update(sample_list)

        # update dbg info
        self.sample_cur_pi.fill(0)
        for x in sample_list:
            self.sample_cur_pi[len(x)] += 1
        self.sample_acc_count += self.sample_cur_pi
        self.sample_cur_pi /= self.sample_cur_pi.sum()

        return None

    def train(self, print_per_epoch=0.1, operation=None):

        # initialize
        self.initialize()

        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]

        print('[TRF] [Train]...')
        time_beginning = time.time()
        model_train_nll = []
        # model_train_nll_phi = []
        model_q_nll = []
        model_kl_dist = []

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

            # draw samples
            with self.time_recoder.recode('sample'):
                sample_seqs = self.draw(self.config.sample_batch_size)

            # update paramters
            with self.time_recoder.recode('update'):
                # learining rate
                self.cur_lr_word = self.config.lr_word.get_lr(step+1, epoch)
                self.cur_lr_logz = self.config.lr_logz.get_lr(step+1, epoch)
                # update
                self.update(data_seqs, sample_seqs)

            # evaulate the nll and KL-distance
            with self.time_recoder.recode('eval_train_nll'):
                nll_train = self.eval(data_seqs)[0]
                nll_q = self.mcmc.eval(data_seqs)[0]
                model_train_nll.append(nll_train)
                # model_train_nll_phi.append(self.eval(data_seqs, is_norm=False)[0])
                model_q_nll.append(nll_q)
                model_kl_dist.append(self.eval(sample_seqs)[0] - self.mcmc.eval(sample_seqs)[0])

            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                time_since_beg = (time.time() - time_beginning) / 60

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                info['lr_word'] = '{:.2e}'.format(self.cur_lr_word)
                info['lr_logz'] = '{:.2e}'.format(self.cur_lr_logz)
                info['lj_rate'] = self.mcmc.lj_rate
                info['mv_rate'] = self.mcmc.mv_rate
                info['train'] = np.mean(model_train_nll[-epoch_step_num:])
                # info['train_phi'] = np.mean(model_train_nll_phi[-100:])
                info['valid'] = model_valid_nll
                info['auxil'] = np.mean(model_q_nll[-epoch_step_num:])
                info['kl_dist'] = np.mean(model_kl_dist[-epoch_step_num:])

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

    def write_log_zeta(self, step, true_logz=None):
        #  write zeta, logz, pi
        f = self.write_files.get('zeta')
        f.write('step={}\n'.format(step))
        log.write_array(f, self.sample_cur_pi[self.config.min_len:], name='cur_pi')
        log.write_array(f, self.sample_acc_count[self.config.min_len:] / self.sample_acc_count.sum(), name='all_pi')
        log.write_array(f, self.config.pi_0[self.config.min_len:], name='pi_0  ')
        log.write_array(f, self.norm_const.get_logz(), name='logz  ')
        if true_logz is not None:
            log.write_array(f, true_logz, name='truez ')

    def get_tag(self, x_list):
        tag_list = []
        logp_list = []
        for x in x_list:
            tag_list.append([0] * len(x))
            logp_list.append(0)
        return tag_list, logp_list


class TRF(crf.CRF):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):

        super().__init__(config, data, logdir, device, name)

        self.pretrain_word = self.config.pretrain_word

        # phi for words
        self.phi_word = pot.create(config.word_config, data.datas[0], config.opt_word, device)

        # logZ
        self.norm_const = norm.Norm(config, data, config.opt_logz)

        # simulater
        # self.propose_lstm = lstmlm.LM(config.auxiliary_config, device=device)
        # length jump probabilities
        self.gamma = sp.len_jump_distribution(self.config.min_len,
                                              self.config.max_len,
                                              self.config.jump_width)

        self.mcmc = mcmc.RJMCMC(q_config=self.config.auxiliary_config,
                                gamma=self.gamma,
                                multiple_trial=self.config.multiple_trial,
                                sample_sub=self.config.sample_sub,
                                end_tokens=self.config.end_tokens,
                                fun_logps=lambda x: self.logps(x, for_eval=False),
                                fun_propose_tag=self.propose_tag if not self.pretrain_word else None,
                                write_files=self.write_files,
                                device=device)

        # sample sequences
        self.mcmc_seqs = [seq.Seq(level=2).random(
                                min_len=self.config.min_len,
                                max_len=self.config.max_len,
                                vocab_sizes=[self.config.word_vocab_size, self.config.tag_vocab_size],
                                beg_tokens=self.config.beg_tokens,
                                end_tokens=self.config.end_tokens,
                                pi=self.config.pi_true)
                            for _ in range(self.config.chain_num)]
        self.mcmc_states = None

        # learning rate
        self.cur_lr_word = 1.0
        self.cur_lr_logz = 1.0

        # debug variables
        # self.lj_times = 1
        # self.lj_success = 0
        # self.lj_rate = 1
        # self.mv_times = 1
        # self.mv_success = 0
        # self.mv_rate = 1
        self.sample_cur_pi = np.zeros(self.config.max_len+1)     # current pi
        self.sample_acc_count = np.zeros(self.config.max_len+1)  # accumulated count

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
            return super().phi(seq_list) if not self.pretrain_word else 0 + \
                   self.phi_word.get_value(seq_list)
        elif filter_name == 'word':
            return self.phi_word.get_value(seq_list)
        elif filter_name == 'tag':
            return self.phi_tag.get_value(seq_list) if not self.pretrain_word else np.zeros(len(seq_list))
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
        if np.any(seq_lens < self.config.min_len) or np.any(seq_lens > self.config.max_len):
            raise TypeError('min_len={}, max_len={} lens={}'.format(min(seq_lens), max(seq_lens), seq_lens))

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

    def logpxs(self, x_list, is_norm=True, for_eval=True):

        logw_h = super().logz(x_list) if not self.pretrain_word else 0

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

    def initialize(self):
        # create features
        super().initialize()

        print('[TRF] init word features.')
        self.phi_word.initialize()

        # print parameters
        print('[TRF] word_feat_num = {:,}'.format(self.phi_word.get_param_num()))

    def propose_tag(self, seq_list, pos_vec, is_sample=True):

        logps = self.get_tag_logps(seq_list, pos_vec)

        if is_sample:
            new_tags = [sp.log_sample(p) for p in logps]
        else:
            new_tags = [s.x[1, i] for s, i in zip(seq_list, pos_vec)]

        cond_logps = logps[np.arange(len(seq_list)), new_tags]

        # # debug ..................
        # logps2 = []
        # for s, pos in zip(seq_list, pos_vec):
        #     save_tag = s.x[1][pos]
        #     logp = np.ones(self.config.tag_vocab_size)
        #     for t in range(self.config.tag_vocab_size):
        #         s.x[1][pos] = t
        #         logp[t] = self.phi_depend_on_tag([s], pos)[0]
        #     s.x[1][pos] = save_tag
        #     logps2.append(sp.log_normalize(logp))
        # # debug ......................
        #
        # diff = logps - np.array(logps2)
        # # print('max_diff', np.max(diff))
        # if np.max(diff) > 1e-6:
        #     raise TypeError('max_diff = {}'.format(np.max(diff)))

        return np.array(new_tags), np.array(cond_logps)

    def sample(self, seq_list, states_list):

        with self.time_recoder.recode('local_jump'):
            seq_list, states_list = self.mcmc.local_jump(seq_list, states_list)

        with self.time_recoder.recode('markov_move'):
            seq_list, states_list = self.mcmc.markov_move(seq_list, states_list)

        # monitor the hidden states

        # true_states = self.mcmc.update_hidden_state(seq_list)
        # d = self.mcmc.hidden_states_diff(true_states, states_list)
        # f = self.write_files.get('mcmc.dbg')
        # f.write('len={} d={}\n'.format([len(s) for s in seq_list], d))
        # f.flush()

        return seq_list, states_list

    def draw(self, n):

        self.mcmc.reset_dbg()

        # init the sample states
        # as at each iteration, the variables of lstm will be updated
        # the hidden states need to be uptated too
        self.mcmc_states = self.mcmc.update_hidden_state(self.mcmc_seqs)

        # draw samples
        res_seq_list = []
        for _ in range(n // len(self.mcmc_seqs)):
            self.mcmc_seqs, self.mcmc_states = self.sample(self.mcmc_seqs, self.mcmc_states)
            res_seq_list += seq.list_copy(self.mcmc_seqs)

        self.mcmc.update_dbg()

        with self.time_recoder.recode('write_sample'):
            f = self.write_files.get('sample')
            for s in res_seq_list:
                f.write(str(s))

        return res_seq_list

    # def update_propose_lstm(self, sample_list):
    #     x_list = [s.x[0].tolist() for s in sample_list]
    #     self.propose_lstm.sequence_update(self.session, x_list)

    def update(self, data_list, sample_list):
        # compute the scalars
        data_scalar = np.ones(len(data_list)) / len(data_list)
        sample_len = np.array([len(x) for x in sample_list])
        sample_facter = np.array(self.config.pi_true[self.config.min_len:]) / \
                        np.array(self.config.pi_0[self.config.min_len:])
        sample_scalar = sample_facter[sample_len - self.config.min_len] / len(sample_list)

        # update phi
        update_phis = []
        if not self.config.fix_trf_model:
            update_phis.append(('word', self.phi_word, self.cur_lr_word))
        if not self.config.fix_crf_model and not self.pretrain_word:
            update_phis.append(('tag', self.phi_tag, self.cur_lr_tag))
            update_phis.append(('mix', self.phi_mix, self.cur_lr_mix))

        for phi_name, phi, lr in update_phis:
            with self.time_recoder.recode('update_' + phi_name):
                phi.update(data_list, data_scalar, sample_list, sample_scalar,
                           learning_rate=lr)

        # update zeta
        with self.time_recoder.recode('update_logz'):
            self.norm_const.update(sample_list, learning_rate=self.cur_lr_logz)
            self.norm_const.set_logz1(self.get_true_logz(self.config.min_len)[0])

        # update simulater
        with self.time_recoder.recode('update_simulater'):
            self.mcmc.update(sample_list)

        # update dbg info
        self.sample_cur_pi.fill(0)
        for x in sample_list:
            self.sample_cur_pi[len(x)] += 1
        self.sample_acc_count += self.sample_cur_pi
        self.sample_cur_pi /= self.sample_cur_pi.sum()

        return None

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
        model_q_nll = []
        model_kl_dist = []

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
                self.update(data_seqs, sample_seqs)

            # evaulate the nll and KL-distance
            with self.time_recoder.recode('eval_train_nll'):
                nll_train = self.eval(data_seqs)[0]
                nll_q = self.mcmc.eval(data_seqs)[0]
                model_train_nll.append(nll_train)
                # model_train_nll_phi.append(self.eval(data_seqs, is_norm=False)[0])
                model_q_nll.append(nll_q)
                model_kl_dist.append(self.eval(sample_seqs)[0] - self.mcmc.eval(sample_seqs)[0])

            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                time_since_beg = (time.time() - time_beginning) / 60

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                info['lr_tag'] = '{:.2e}'.format(self.cur_lr_tag)
                info['lr_mix'] = '{:.2e}'.format(self.cur_lr_mix)
                info['lr_word'] = '{:.2e}'.format(self.cur_lr_word)
                info['lr_logz'] = '{:.2e}'.format(self.cur_lr_logz)
                info['lj_rate'] = self.mcmc.lj_rate
                info['mv_rate'] = self.mcmc.mv_rate
                info['train'] = np.mean(model_train_nll[-epoch_step_num:])
                # info['train_phi'] = np.mean(model_train_nll_phi[-100:])
                info['valid'] = model_valid_nll
                info['auxil'] = np.mean(model_q_nll[-epoch_step_num:])
                info['kl_dist'] = np.mean(model_kl_dist[-epoch_step_num:])

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

    def write_log_zeta(self, step, true_logz=None):
        #  write zeta, logz, pi
        f = self.write_files.get('zeta')
        f.write('step={}\n'.format(step))
        log.write_array(f, self.sample_cur_pi[self.config.min_len:], name='cur_pi')
        log.write_array(f, self.sample_acc_count[self.config.min_len:] / self.sample_acc_count.sum(), name='all_pi')
        log.write_array(f, self.config.pi_0[self.config.min_len:], name='pi_0  ')
        log.write_array(f, self.norm_const.get_logz(), name='logz  ')
        if true_logz is not None:
            log.write_array(f, true_logz, name='truez ')

    def debug_get_logpx(self):
        self.initialize()
        self.save()

        print('tag_order=', self.phi_tag.get_order())

        for set_value in [1]:
            # init the values
            if isinstance(self.phi_word_feat, pot.FeatPhi):
                self.phi_word_feat.feats.values = np.random.uniform(size=self.phi_word_feat.get_param_num()) * set_value

            self.phi_tag.feats.values = np.random.uniform(size=self.phi_tag.get_param_num()) * set_value
            self.phi_tag.need_update_trans_matrix = True

            self.phi_mix.feats.values = np.random.uniform(size=self.phi_mix.get_param_num()) * set_value

            for seq_len in [3, 4]:

                x_list = [sp.random_seq(seq_len, seq_len,
                                        self.config.word_vocab_size,
                                        self.config.beg_tokens[0],
                                        self.config.end_tokens[0]) for _ in range(2)]

                s_list = [seq.Seq(x) for x in x_list]

                logpx1 = self.get_logpxs(x_list, is_norm=False)

                logpx_list = []
                tag_iter = sp.SeqIter(seq_len,
                                      self.config.tag_vocab_size,
                                      self.config.beg_tokens[1],
                                      self.config.end_tokens[1])
                for tags in tag_iter:
                    for s in s_list:
                        s.x[1] = tags
                    logpx_list.append(self.phi(s_list))
                logpx2 = logsumexp(logpx_list, axis=0)
                print('v={}, len={}, logpx1={}, logpx2={}'.format(set_value, seq_len, logpx1, logpx2))

    def debug_logz(self):
        seq_len = 4

        logz1 = self.get_true_logz(3)
        self.norm_const.set_logz1(logz1)

        seq_list = []
        for w in sp.SeqIter(seq_len, self.config.word_vocab_size, beg_token=self.config.beg_tokens[0],
                            end_token=self.config.end_tokens[0]):
            for t in sp.SeqIter(seq_len, self.config.tag_vocab_size, beg_token=self.config.beg_tokens[1],
                                end_token=self.config.end_tokens[1]):
                s = seq.Seq([w, t])
                seq_list.append(s)

        logz2 = logsumexp(self.get_log_probs(seq_list, is_norm=False))

        # print(np.sum(self.config.pi_true))
        print('cur_logz={}, true_logz={} logpi={}'.format(self.norm_const.get_logz(seq_len), logz2, np.log(self.config.pi_true[seq_len])))


    def debug_test_sample(self):

        # init the values
        if isinstance(self.phi_word_feat, pot.FeatPhi):
            self.phi_word_feat.feats.values = np.random.uniform(size=self.phi_word_feat.get_param_num()) * 1

        self.phi_tag.feats.values = np.random.uniform(size=self.phi_tag.get_param_num()) * 1
        self.phi_tag.need_update_trans_matrix = True

        self.phi_mix.feats.values = np.random.uniform(size=self.phi_mix.get_param_num()) * 1

        # normalize
        self.norm_const.set_logz(self.get_true_logz())
        print(self.norm_const.get_logz())

        time_beg = time.time()
        # summary the length distribution
        sample_pi_all = np.zeros(self.config.max_len+1)
        sample_pi_cur = np.zeros(self.config.max_len+1)

        print('begin sampling')
        batch_num = 100
        batch_size = 100
        for t in range(batch_num):
            sample_pi_cur.fill(0)

            sample_list = self.draw(batch_size)

            for s in sample_list:
                sample_pi_cur[len(s)] += 1

            sample_pi_all += sample_pi_cur

            print('sample_pi=', sample_pi_all[self.config.min_len:] / np.sum(sample_pi_all))
            print('true_pi_0=', self.config.pi_0[self.config.min_len:])
            time_since_beg = time.time() - time_beg
            print('t={} time={:.2f} time_per_step={:.2f}'.format(t, time_since_beg, time_since_beg/(t+1)))


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

