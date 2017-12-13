import tensorflow as tf
import numpy as np
import time
import json
import os
from copy import deepcopy
from collections import OrderedDict

from base import *


def array2str(a):
    s = '[ '
    for x in a:
        s += '{:.4f} '.format(x)
    s += ']'
    return s


def to_str(v, fmt='{:.3f}'):
    if isinstance(v, int):
        return str(v)
    elif isinstance(v, float):
        return fmt.format(v)
    elif isinstance(v, list) or isinstance(v, tuple):
        return '[' + ','.join([fmt.format(i) for i in v]) + ']'
    elif isinstance(v, np.ndarray):
        if v.ndim == 0:
            return fmt.format(float(v))
        else:
            return '[' + ','.join([fmt.format(i) for i in v.flatten()]) + ']'
    else:
        return str(v)


def print_line(info, end=' '):
    for name, v in info.items():
        print(name + '=' + to_str(v), end=end, flush=True)


class LearningRate(object):
    def __init__(self):
        pass

    def get_lr(self, t, epoch=None):
        return 1.0

    def __str__(self):
        return 'learning-rate-base-class'


class LearningRateSegment(LearningRate):
    def __init__(self, seg):
        super().__init__()
        self.seg = seg


class LearningRateDelay(LearningRate):
    def __init__(self, init, delay, delay_when, delay_per_iter):
        super().__init__()
        self.init = init
        self.delay = delay
        self.delay_when = delay_when
        self.delay_per_iter = delay_per_iter

    def get_lr(self, t, epoch=None):
        return self.init * self.delay ** (max(0, t - self.delay_when)//self.delay_per_iter)

    def __str__(self):
        return 'init={} delay={} when={} per={}'.format(
            self.init, self.delay, self.delay_when, self.delay_per_iter
        )


class LearningRateTime(LearningRate):
    def __init__(self, a=1., beta=0., t0=None, lr_min=0., lr_max=1000., tc=0):
        super().__init__()
        self.a = a
        self.beta = beta
        self.t0 = t0
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.tc = tc

    def get_lr(self, t, epoch=None):
        """
        calculate current learning rate
        :param t: the iteration number, from 1, to t_max
        :return: return the learining rate:
            if t <= t0: retrun a / t^beta
            if t > t0:  return a / (t0^beta + (t-t0))
        """
        if self.t0 is None or t <= self.t0:
            lr = 1.0 * self.a / (self.tc + t ** self.beta)
        else:
            lr = 1.0 * self.a / (self.tc + self.t0 ** self.beta + t - self.t0)

        return np.clip(lr, a_min=self.lr_min, a_max=self.lr_max)

    def __str__(self):
        return 'a={} beta={} t0={} min={} max={} tc={}'.format(
            self.a, self.beta, self.t0, self.lr_min, self.lr_max, self.tc)

    def write(self, f, name):
        f.write('{} = [ a: {} beta: {:.4f} t0: {} lr_min: {} lr_max: {} tc: {} ]\n'.format(
            name, self.a, self.beta, self.t0, self.lr_min, self.lr_max, self.tc))

    def read(self, f):
        line = f.next()
        a = line.split()
        self.a = float(a[4])
        self.beta = float(a[6])
        if a[8].lower() == 'none':
            self.t0 = None
        else:
            self.t0 = int(a[8])
        self.lr_min = float(a[10])
        return self


class LearningRateEpochDelay(LearningRate):
    def __init__(self, init, delay=1.0, delay_when=0, per_epoch=1):
        super().__init__()
        self.init = init
        self.delay = delay
        self.delay_when = delay_when
        self.per_epoch = per_epoch

    def get_lr(self, t, epoch=0):
        return self.init * self.delay ** (max(0, (int(epoch) - self.delay_when)//self.per_epoch))

    def __str__(self):
        return 'init={} delay={} when={}'.format(
            self.init, self.delay, self.delay_when
        )


class BaseConfig(wb.Config):
    def __init__(self, data):
        self.min_len = data.get_min_len()
        self.max_len = data.get_max_len()
        self.vocab_size = data.get_vocab_size()
        self.pi_true = data.get_pi_true()
        self.pi_0 = data.get_pi0()
        self.beg_token = data.get_beg_token()
        self.end_token = data.get_end_token()
        # for discrete features
        self.feat_type_file = None
        self.feat_cluster = None

        # init zeta
        self.init_zeta = self.get_initial_logz()

    # def value2str(self, v):
    #     if isinstance(v, LearningRate):
    #         return str(v)
    #     return super().value2str(v)

    def get_initial_logz(self, c=None):
        if c is None:
            c = np.log(self.vocab_size)
        len_num = self.max_len - self.min_len + 1
        logz = c * (1 + np.linspace(1, len_num, len_num))
        return logz - logz[0]


class TRFFrame(object):
    def __init__(self, config, data, logdir='trflogs', name='trf'):
        self.config = config
        self.data = data  # the data instance of reader.Data()
        self.logdir = logdir
        self.name = name

        # normalization constants
        len_num = self.config.max_len - self.config.min_len + 1
        self.logz = np.append(np.zeros(self.config.min_len),
                              np.log(self.config.vocab_size) * np.linspace(1, len_num, len_num))
        self.zeta = self.logz - self.logz[self.config.min_len]
        self.gamma = sp.len_jump_distribution(self.config.min_len,
                                              self.config.max_len,
                                              self.config.jump_width)  # length jump probabilities

        # sample sequences
        self.sample_seq = reader.produce_data_to_trf(
            [sp.random_seq(self.config.min_len,
                           self.config.max_len,
                           self.config.vocab_size,
                           beg_token=self.config.beg_token,
                           end_token=self.config.end_token,
                           pi=self.config.pi_true)
             for _ in range(self.config.chain_num)])
        # save the adjacent mcmc state
        # i.e. the adjacent mcmc sequences
        self.sample_mcmc = []

        # summary
        self.time_recoder = wb.clock()
        self.lj_times = 1
        self.lj_success = 0
        self.lj_rate = 1
        self.mv_times = 1
        self.mv_success = 0
        self.mv_rate = 1
        self.sample_cur_pi = np.zeros(self.config.max_len+1)     # current pi
        self.sample_acc_count = np.zeros(self.config.max_len+1)  # accumulated count

        # discrete features
        self.feat_word = None
        self.feat_class = None
        if config.feat_type_file is not None:
            w_type, c_type = trfngram.separate_type(
                trfngram.read_feattype_file(config.feat_type_file)
            )
            if config.opt_method.lower() == 'adam':
                opt_config = {'name': 'adam'}
            else:
                opt_config = {'name': 'var', 'vgap': config.var_gap}
            self.feat_word = trfngram.feat(w_type, opt_config)
            if data.word_to_class is not None:
                self.feat_class = trfngram.feat(c_type, opt_config)

        # output files
        wb.mkdir(logdir)
        self.write_files = wb.FileBank(default_file_name=os.path.join(logdir, name))

        # self.write_pi = open(log_full_path + '.debug', 'wt')
        # self.write_sample = open(log_full_path + '.sample', 'wt')
        # self.write_train = open(log_full_path + '.train', 'wt')
        # self.write_mv = open(log_full_path + '.markov', 'wt')
        # self.write_time = open(log_full_path + '.time', 'wt')

    def phi(self, inputs, lengths):
        """
        given a batch of input sequence and their lengths,
        compute the energy function phi.

        Args:
            inputs: np.array of shape [batch_size, max_length],
                    indicating the algined sequences
            lengths: np.array of shape [batch_size]
                    indicating the length of each sequence

        Returns:
            a np.array of shape [batch_size]
            indicating the phi of each sequence
        """
        feat_weight = np.zeros(len(inputs), dtype='float32')
        seq_list = reader.extract_data_from_trf(inputs, lengths)
        if self.feat_word is not None:
            feat_weight += self.feat_word.seq_list_weight(seq_list)
        if self.feat_class is not None:
            feat_weight += self.feat_class.seq_list_weight(self.data.seqs_to_class(seq_list))
        return feat_weight

    def logps(self, inputs, lengths):
        """
        given a batch of input sequence and their lengths,
        compute the log probs of a sequence.

        Args:
            inputs: np.array of shape [batch_size, max_length],
                    indicating the algined sequences
            lengths: np.array of shape [batch_size]
                    indicating the length of each sequence

        Returns:
            a np.array of shape [batch_size]
            indicating the logprob of each sequence
        """
        return self.phi(inputs, lengths) + \
               np.log(self.config.pi_0[lengths]) - \
               self.logz[lengths]

    def save_feat(self, logname):
        """
        write the discrete ngram features to files
            write word-feat to dir/model.wfeat
            write class-feat to dir/model.cfeat
        """
        def pp(s):
            print('[TRF] save features to', s)

        if self.feat_word is not None:
            with open(logname + '.wfeat', 'wt') as f:
                pp(logname + '.wfeat')
                self.feat_word.write(f)
        if self.feat_class is not None:
            with open(logname + '.cfeat', 'wt') as f:
                pp(logname + '.cfeat')
                self.feat_class.write(f)

    def load_feat(self, logname):
        """
        read word-feat form dir/model.wfeat
        read class-feat form dir/model.cfeat
        """
        def pp(s):
            print('[TRF] load features from', s)

        if self.feat_word is not None and \
                wb.exists(logname + '.wfeat'):
            with open(logname + '.wfeat', 'rt') as f:
                pp(logname + '.wfeat')
                self.feat_word.read(f)
        if self.feat_class is not None and \
                wb.exists(logname + '.cfeat'):
            with open(logname + '.cfeat', 'rt') as f:
                pp(logname + '.cfeat')
                self.feat_class.read(f)

    def save_model(self, logname):

        def write_value(f, v, label):
            try:
                f.write('{} = {}\n'.format(label, json.dumps(v)))
            except TypeError:
                print('value type =', type(v))

        print('[TRF] save model info to', logname + '.info')
        with open(logname + '.info', 'wt') as f:
            write_value(f, int(self.config.max_len), 'max_len')
            write_value(f, int(self.config.min_len), 'min_len')
            write_value(f, self.logz.tolist(), 'logz')
            write_value(f, self.zeta.tolist(), 'zeta')

    def load_model(self, logname):

        def read_value(f, label):
            s = f.readline()
            idx = s.find(' = ')
            if s[0: idx] != label:
                print('[E] read_value label=[{}] error, current label=[{}]\n at line {}'.format(label, s[0: idx], s))
                raise IOError('Face an error label=[{}], request label=[{}]'.format(s[0: idx], label))
            return json.loads(s[idx+2:])

        if not wb.exists(logname + '.info'):
            return

        print('[TRF] load model info from', logname + '.info')
        with open(logname + '.info', 'rt') as f:
            assert read_value(f, 'max_len') == self.config.max_len
            assert read_value(f, 'min_len') == self.config.min_len
            self.logz = np.array(read_value(f, 'logz'))
            self.zeta = np.array(read_value(f, 'zeta'))

    def precompute_feat(self, train_list):
        """
        prapare features before training. The operations include:
            1. if feature number == 0, exact features from train_list;
            2. compute the empirical expectation on dataset train_list;
            3. write the expectation to files
        """
        tbeg = time.time()
        if self.feat_word is not None:
            if self.feat_word.num == 0:
                self.feat_word.load_from_seqs(train_list)
            print('[TRF] pre-compute word-feat...')
            self.feat_word.precompute_seq_train_expec(train_list, 1.0 / len(train_list),
                                                      logname=os.path.join(self.logdir, self.name + '.wfeat'))
        if self.feat_class is not None:
            train_class_list = self.data.seqs_to_class(train_list)
            if self.feat_class.num == 0:
                self.feat_class.load_from_seqs(train_class_list)
            print('[TRF] pre-compute class-feat...')
            self.feat_class.precompute_seq_train_expec(train_class_list, 1.0 / len(train_list),
                                                       logname=os.path.join(self.logdir, self.name + '.cfeat'))
        print('[TRF] finished (time={:.2f})'.format((time.time() - tbeg) / 60))

    def get_log_probs(self, seq_list, is_norm=True):
        """
        Calling self.phi to calculate the log probs of each sequences
        Args:
            seq_list: a list of sequences
            is_norm: if True, then normalize the log probs.

        Returns:
            a np.array of shape [batch_size],
            indicating the log probs of each sequence
        """
        batch_size = self.config.batch_size
        logprobs = np.zeros(len(seq_list))
        for i in range(0, len(seq_list), batch_size):
            logprobs[i: i+batch_size] = self.phi(
                *reader.produce_data_to_trf(seq_list[i: i+batch_size])
            )

        if is_norm:
            n = [len(x) for x in seq_list]
            logprobs += np.log(self.config.pi_0[n]) - self.logz[n]
        return logprobs

    def get_log_prob(self, seq, is_norm=True):
        """
        Calling self.get_log_probs to compute the log prob of one sequence
        Args:
            seq: a list, indicating the sequence
            is_norm: if True, then normalize the log probs

        Returns:
            a float, the log prob of given sequence
        """
        return self.get_log_probs([seq], is_norm)[0]

    def true_normalize(self, length):
        """
        Calling self.get_log_probs to compute the normalization constant,
        using the Naive method.
        Args:
            length: the length

        Returns:
            the log normalization constant
        """
        assert length <= self.config.max_len
        assert length >= self.config.min_len
        # simple way
        x_batch = [x for x in sp.SeqIter(length, self.config.vocab_size, self.config.beg_token, self.config.end_token)]
        logz = sp.log_sum(self.get_log_probs(x_batch, False))
        return logz

    def true_normalize_all(self):
        """
        Calling self.true_normalize to compute the normalization constants
        for all lengths

        Returns:
            None
        """
        for l in range(self.config.min_len, self.config.max_len+1):
            self.logz[l] = self.true_normalize(l)
        self.zeta = self.logz - self.logz[self.config.min_len]

    def get_marginal_log_prob(self, seq, pos):
        """
        Calling self.get_log_probs to compute the marginal log prob
        used to sample.

        Args:
            seq: a list indicating the sequence
            pos: the position

        Returns:
            a np.array denoting the 1-d probs
        """
        input_x = np.array(seq, dtype='int32')
        input_x = np.tile(input_x, (self.config.vocab_size, 1))
        input_x[:, pos] = np.arange(self.config.vocab_size)
        logprobs = self.get_log_probs(input_x, is_norm=False)
        return sp.log_normalize(logprobs)

    def local_jump(self, x):
        """
        Calling self.get_marginal_log_prob, self.get_log_probs to
        perform local jump to change the length of input sequences

        Args:
            x: a list indicating the sequence

        Returns:
            a net list
        """
        k = len(x)  # old length
        j = sp.linear_sample(self.gamma[k])  # new length

        assert(j >= self.config.min_len)
        assert(j <= self.config.max_len)

        if j == k:
            return x

        # local jump
        if j > k:
            # generate a new sequence new_x
            new_x = x[0:]  # copy list
            g_add = 0
            while len(new_x) < j:
                new_x.insert(-1, 0)  # add a new position.
                g = np.exp(self.get_marginal_log_prob(new_x, len(new_x)-2))
                new_x[-2] = sp.linear_sample(g)  # sample the last position
                g_add += np.log(g[new_x[-2]])

            # calculate the probabilities and the acceptance probability
            p_new = self.get_log_prob(new_x)
            p_old = self.get_log_prob(x)
            acc_logp = np.log(self.gamma[j, k]) - np.log(self.gamma[k, j]) + p_new - p_old - g_add

        elif j < k:
            # generate a new sequence new_x
            new_x = x[0:]
            g_add = 0
            while len(new_x) > j:
                g = np.exp(self.get_marginal_log_prob(new_x, len(new_x)-2))
                g_add += np.log(g[new_x[-2]])
                del new_x[-2]

            # calculate the probabilities and the acceptance probability
            p_new = self.get_log_prob(new_x)
            p_old = self.get_log_prob(x)
            acc_logp = np.log(self.gamma[j, k]) - np.log(self.gamma[k, j]) + p_new + g_add - p_old

        else:
            raise Exception('Invalid jump form {} to {}'.format(k, j))

        self.lj_times += 1
        if sp.accept(np.exp(acc_logp)):
            self.lj_success += 1
            x = new_x

        out_line = '[local jump] {}->{} acc_prob={:.2f} '.format(k, j, np.exp(float(acc_logp)))
        out_line += 'p_new={:.2f} p_old={:.2f} '.format(float(p_new), float(p_old))
        out_line += 'g_add={:.2f} '.format(float(g_add))
        out_line += 'acc_rate={:.2f}% '.format(100.0 * self.lj_success / self.lj_times)
        out_line += '[{}/{}] '.format(self.lj_success, self.lj_times)
        f = self.write_files.get('markov')
        f.write(out_line + '\n')
        f.flush()

        return x

    def markov_move(self, x):
        """
        perform gibbs sampling

        Args:
            x: a list indicating the sequence

        Returns:
            a new list
        """
        head = 0
        tail = len(x)
        for pos in range(head, tail):
            g = self.get_marginal_log_prob(x, pos)
            x[pos] = sp.log_sample(g)

        return x

    def sample(self, input_x, input_n):
        """
        perform local_jump and markov_move

        Args:
            input_x: np.array of shape [batch_size, max_length],
                    indicating the algined sequences
            input_n: np.array of shape [batch_size]
                    indicating the length of each sequence

        Returns:
            a tuple ( new_input_x, new_input_n )
        """
        with self.time_recoder.recode('local_jump'):
            input_x, input_n = reader.produce_data_to_trf(
                [self.local_jump(x) for x in reader.extract_data_from_trf(input_x, input_n)]
            )

        with self.time_recoder.recode('markov_move'):
            input_x, input_n = reader.produce_data_to_trf(
                [self.markov_move(x) for x in reader.extract_data_from_trf(input_x, input_n)]
            )

        return input_x, input_n

    def draw(self, n):
        """
        calling self.sample to draw n samples

        Args:
            n: the sample numbers

        Returns:
            a list of n sequences
        """
        self.lj_times = 0
        self.lj_success = 0
        self.mv_times = 0
        self.mv_success = 0
        self.sample_mcmc = [(np.array(self.sample_seq[0]), np.array(self.sample_seq[1]))]

        seq_list = []
        for i in range(n//self.config.chain_num):
            self.sample_seq = self.sample(*self.sample_seq)
            seq_list += reader.extract_data_from_trf(*self.sample_seq)   # copy the sequence
            self.sample_mcmc.append((np.array(self.sample_seq[0]), np.array(self.sample_seq[1])))

        if self.lj_times > 0:
            self.lj_rate = 0.9 * self.lj_rate + 0.1 * (self.lj_success / self.lj_times)
        if self.mv_times > 0:
            self.mv_rate = 0.9 * self.mv_rate + 0.1 * (self.mv_success / self.mv_times)

        # with self.time_recoder.recode('write_sample'):
        #     f = self.write_files.get('sample')
        #     f.write('\n'.join([str(a) for a in seq_list]) + '\n')
        #     f.flush()
        #
        #     f = self.write_files.get('mcmc')
        #     for x, n in self.sample_mcmc:
        #         seqs = reader.extract_data_from_trf(x, n)
        #         f.write('\n'.join([str(a) for a in seqs]) + '\n')
        #     f.flush()

        return seq_list

    def update_zeta(self, sample_list, lr_zeta):
        """
        Given the sample list, update zeta

        Args:
            sample_list: a list of sequences
            lr_zeta: learning rate of zeta

        Returns:
            None
        """

        # update zeta
        sample_pi = np.zeros(self.config.max_len+1)
        for seq in sample_list:
            sample_pi[len(seq)] += 1.
        sample_pi /= len(sample_list)
        zeta_step = np.clip(lr_zeta * sample_pi[self.config.min_len:] / self.config.pi_0[self.config.min_len:],
                            a_min=0, a_max=self.config.zeta_gap)
        self.zeta[self.config.min_len:] += zeta_step
        self.zeta[self.config.min_len:] -= self.zeta[self.config.min_len]
        self.logz[self.config.min_len:] = self.zeta[self.config.min_len:] + self.true_normalize(self.config.min_len)
        self.sample_acc_count += sample_pi * len(sample_list)
        self.sample_cur_pi = sample_pi

    def eval(self, data_list):
        """
        compute the negative log-likelihood and PPL on data set,
        using the pi_true

        Args:
            data_list: a list of sequences

        Returns:
            tuple( neg-log-likelihood avarged over sentences, ppl )
        """
        pi_0_save = self.config.pi_0
        self.config.pi_0 = self.config.pi_true
        logps = self.get_log_probs(data_list)
        self.config.pi_0 = pi_0_save

        lens = [len(x) - int(self.config.beg_token is not None) for x in data_list]
        s = - sum(logps)
        nll = s / len(data_list)
        ppl = np.exp(s/sum(lens))
        return nll, ppl

    def eval_pi0(self, data_list):
        """
        compute the negative log-likelihood and PPL on data set,
        using the pi_0

        Args:
            data_list: a list of sequences

        Returns:
            tuple( neg-log-likelihood avarged over sentences, ppl )
        """
        logps = self.get_log_probs(data_list)

        lens = [len(x) - int(self.config.beg_token is not None) for x in data_list]
        s = - sum(logps)
        nll = s / len(data_list)
        ppl = np.exp(s / sum(lens))
        return nll, ppl

    def logwrite_pi(self, logz_sams=None, logz_true=None, step=None):
        def atos(a):
            return ' '.join(['{:.5f}'.format(p) for p in a[self.config.min_len:]]) + '\n'

        f = self.write_files.get('zeta')
        if step is not None:
            f.write('step={}\n'.format(step))
        f.write('cur_pi= ' + atos(self.sample_cur_pi))
        f.write('all_pi= ' + atos(self.sample_acc_count/self.sample_acc_count.sum()))
        f.write('pi_0  = ' + atos(self.config.pi_0))
        f.write('zeta  = ' + atos(self.zeta))
        if logz_sams is not None:
            f.write('logz  = ' + atos(logz_sams))
        if logz_true is not None:
            f.write('logz* = ' + atos(logz_true) + '\n')
        f.flush()

    def pi_distance(self):
        sample_pi = self.sample_acc_count / self.sample_acc_count.sum()
        diff = sample_pi - self.config.pi_0
        return np.sqrt(np.sum(diff[self.config.min_len:] ** 2))

    def test_sample(self):
        """
        Test the sample method
        """
        self.true_normalize_all()
        print(self.logz[self.config.min_len:])

        # compute the marginal probability
        time_beg = time.time()
        comp_len = 3
        comp_pos = 1
        x_batch = [x for x in sp.SeqIter(comp_len, self.config.vocab_size, self.config.beg_token, self.config.end_token)]
        x_logp = self.get_log_probs(x_batch, True)
        print('x_logp', x_logp)
        local_p = np.zeros(self.config.vocab_size)
        for x, logp in zip(x_batch, x_logp):
            local_p[x[comp_pos]] += np.exp(logp)
        sample_local_p = np.zeros(self.config.vocab_size)

        # summary the length distribution
        sample_pi_all = np.zeros(self.config.max_len+1)
        sample_pi_cur = np.zeros(self.config.max_len+1)

        print('begin sampling:')
        batch_num = 100
        batch_size = 100
        for t in range(batch_num):
            sample_pi_cur.fill(0)

            sample_list = self.draw(batch_size)

            for s in sample_list:
                sample_pi_cur[len(s)] += 1

                if len(s) == comp_len:
                    sample_local_p[s[comp_pos]] += 1

            sample_pi_all += sample_pi_cur

            f = self.write_files.get('sample')
            for s in sample_list:
                f.write(' '.join([str(w) for w in s]) + '\n')
            f.flush()

            print('sample_pi=', sample_pi_all[self.config.min_len:] / np.sum(sample_pi_all))
            print('true_pi_0=', self.config.pi_0[self.config.min_len:])
            print('sample_p=', sample_local_p / np.sum(sample_pi_all))
            print('trup_p  =', local_p)
            time_since_beg = time.time() - time_beg
            print('t={} time={:.2f} time_per_step={:.2f}'.format(t, time_since_beg, time_since_beg/(t+1)))



















