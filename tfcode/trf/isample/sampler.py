import tensorflow as tf
import copy
from base import *

from trf.nce import noise
from . import lstm


def create_sampler(trf_config, data, device='/gpu:0', limit_vocab=None):
    if isinstance(trf_config.sampler_config, LSTM.Config):
        sampler = LSTM(trf_config, device)
    elif isinstance(trf_config.sampler_config, LSTMLen.Config):
        sampler = LSTMLen(trf_config, device)
    elif isinstance(trf_config.sampler_config, Ngram.Config):
        sampler = Ngram(trf_config, data)
    else:
        raise TypeError('[{}] undefined sampler_config type = {}'.format(__name__, type(trf_config.sampler_config)))

    if limit_vocab is not None and limit_vocab < trf_config.vocab_size:
        print('[{}] create limit_vocab_sampler'.format(__name__))
        return LimitVocab(sampler, data, limit_vocab)
    else:
        return sampler


class Base(object):
    def __init__(self, trf_config):
        self.config = trf_config

    def initialize(self):
        pass

    def generate(self, num):
        pass

    def get_log_probs(self, sample_list):
        return np.zeros(len(sample_list))

    def eval_nll(self, sample_list):
        logps = self.get_log_probs(sample_list)
        nll = -np.mean(logps)
        return nll

    def eval_ppl(self, sample_list):
        logps = self.get_log_probs(sample_list)
        words = np.sum([len(x) - 1 for x in sample_list])
        ppl = np.exp(-np.sum(logps) / words)
        return ppl

    def update(self, sample_list, scale, lr=None):
        pass

    def save(self, fname):
        pass

    def restore(self, fname):
        pass


def data_reduce_vocab(seq_list, limit_vocab):
    res_list = copy.deepcopy(seq_list)
    for s in res_list:
        for i in range(len(s)):
            if s[i] >= limit_vocab:
                s[i] = limit_vocab - 1
    return res_list


class LimitVocab(Base):
    def __init__(self, base_sampler, data, limit_vocab=10000):
        super().__init__(base_sampler.config)
        self.base_sampler = base_sampler

        self.vocab_unigram = data.get_unigram()
        self.limit_vocab_size = limit_vocab
        self.vocab_size = data.get_vocab_size()
        self.extra_vocab_unigram = self.vocab_unigram[self.limit_vocab_size:]
        self.extra_vocab_unigram /= np.sum(self.extra_vocab_unigram)
        self.extra_vocab_unigram = sp.FastProb(self.extra_vocab_unigram)

        if self.limit_vocab_size < self.vocab_size:
            print('[{}.{}] for large vocabulary!'.format(__name__, self.__class__.__name__))
            print('[{}.{}] extra_vocab_unigram size={}'.format(__name__, self.__class__.__name__, self.extra_vocab_unigram.size))

    def data_reduce_vocab(self, seq_list):
        if self.limit_vocab_size == self.vocab_size:
            return seq_list
        return data_reduce_vocab(seq_list, self.limit_vocab_size)

    def data_extend_vocab(self, seq_list):
        if self.limit_vocab_size == self.vocab_size:
            return seq_list

        for s in seq_list:
            for i in range(len(s)):
                if s[i] == self.limit_vocab_size - 1:
                    s[i] += self.extra_vocab_unigram.sample()
        return seq_list

    def get_extra_logp(self, seq_list):
        if self.limit_vocab_size == self.vocab_size:
            return 0

        logps_extra = []
        for s in seq_list:
            logp = 0
            for i in range(len(s)):
                if s[i] >= self.limit_vocab_size:
                    logp += np.log(self.extra_vocab_unigram[s[i] - self.limit_vocab_size])
            logps_extra.append(logp)
        return np.array(logps_extra)

    def initialize(self):
        self.base_sampler.initialize()

    def get_log_probs(self, sample_list):
        logp = self.base_sampler.get_log_probs(self.data_reduce_vocab(sample_list))
        return logp + self.get_extra_logp(sample_list)

    def generate(self, num):
        sample_list = self.base_sampler.generate(num)
        return self.data_extend_vocab(sample_list)

    def update(self, sample_list, scale, lr=None):
        return self.base_sampler.update(self.data_reduce_vocab(sample_list), scale, lr)

    def save(self, fname):
        self.base_sampler.save(fname)

    def restore(self, fname):
        self.base_sampler.restore(fname)


class LSTM(Base):
    class Config(lstm.Config):
        def __str__(self):
            return 'LSTMGen'

    def __init__(self, trf_config, device):
        """
        :param trf_config:
        :param device: str or a list of str, such as '/gpu:0' or ['/gpu:0', '/gpu:1']
        """
        super().__init__(trf_config)
        self.len_prob = trf_config.pi_0
        self.lstm = lstm.Model(trf_config.sampler_config, devices=device, name='auxiliary_lstm')

    def generate(self, num):
        sample_list = []

        batch_size = 128
        while len(sample_list) < num:
            wlist = self.lstm.draw_seqs(tf.get_default_session(), batch_size,
                                        self.config.beg_token, self.config.end_token,
                                        max_length=self.config.max_len)

            for w in wlist:
                if len(w) >= self.config.min_len:
                    sample_list.append(w)

        sample_list = sample_list[0:num]
        return sample_list

    def get_log_probs(self, sample_list):
        return self.lstm.get_log_probs(tf.get_default_session(), sample_list, 100)

    def update(self, sample_list, scale, lr=None, batch_size=100):
        for i in range(0, len(sample_list), batch_size):
            self.lstm.update(tf.get_default_session(), sample_list[i:i+batch_size], scale[i:i+batch_size], lr)

    def save(self, fname):
        self.lstm.save(tf.get_default_session(), fname)

    def restore(self, fname):
        self.lstm.restore(tf.get_default_session(), fname)

    def train(self, train_list, valid_list, batch_size,
                    lr, max_epoch=10,
                    print_per_epoch=0.1, operation=None,
                    write_model=None):

        total_num = len(train_list)
        total_batch = total_num // batch_size

        for epoch in range(max_epoch):
            np.random.shuffle(train_list)
            nll = []

            for i in range(total_batch):

                cur_step = total_batch * epoch + i + 1
                cur_epoch = cur_step / total_batch

                data_list = train_list[i * batch_size: i * batch_size + batch_size]
                self.update(data_list,
                            np.ones(batch_size) / batch_size,
                            lr=lr.get_lr(cur_step, cur_epoch))
                nll.append(self.eval_nll(data_list))

                if cur_step % (int(print_per_epoch * total_batch)) == 0:
                    print('step={} epoch={:.2f} train_nll={:.2f}'.format(
                        cur_step, cur_epoch, np.mean(nll[-total_batch:]))
                        )

            valid_nll = self.eval_nll(valid_list)
            print('*** epoch={} valid_nll={:.2f}'.format(epoch + 1, valid_nll))

            if operation is not None:
                operation.run((epoch + 1) * total_batch, epoch+1)

            if write_model is not None:
                print('*** epoch={} write to {}'.format(epoch+1, write_model))
                self.save(write_model)


class LSTMLen(Base):
    class Config(lstm.Config):
        def __init__(self, data_or_vocab_size=1000, hidden_size=200, hidden_layers=2):
            super().__init__(data_or_vocab_size, hidden_size, hidden_layers)
            self.max_batch_per_core = 128

        def __str__(self):
            return 'LSTMLenGen'

    def __init__(self, trf_config, device='/gpu:0'):
        super().__init__(trf_config)

        self.sampler_config = trf_config.sampler_config
        self.len_prob = np.array(trf_config.pi_0)

        # SUPPORT: trf_config.sampler_config.vocab_size <= trf_config.vocab_size
        self.device_num = 1 if isinstance(device, str) else len(device)

        self.lstm = lstm.Model(trf_config.sampler_config, devices=device, name='auxiliary_lstm_len')

    def generate(self, num):
        sample_list = []

        lengths = np.random.choice(len(self.len_prob), size=num, p=self.len_prob)
        assert np.all(lengths >= self.config.min_len)

        wlist = self.lstm.draw_seqs(tf.get_default_session(), num,
                                    self.config.beg_token, None,
                                    max_length=np.max(lengths))

        for w, l in zip(wlist, lengths):
            w = w[0: l]
            w[-1] = self.config.end_token
            sample_list.append(w)

        # batch_size = self.sampler_config.max_batch_per_core * self.device_num
        # while len(sample_list) < num:
        #     lengths = np.random.choice(len(self.len_prob), size=batch_size, p=self.len_prob)
        #
        #     max_len = np.max(lengths)
        #     wlist = self.lstm.draw_seqs(tf.get_default_session(), batch_size,
        #                                 self.config.beg_token, None,
        #                                 max_length=max_len)
        #     for w, l in zip(wlist, lengths):
        #         w = w[0: l]
        #         w[-1] = self.config.end_token
        #         if len(w) >= self.config.min_len:
        #             sample_list.append(w)
        #
        # sample_list = sample_list[0:num]
        return sample_list

    # def add_noise(self, seq_list):
    #     seqs = self.lstm.add_noise(tf.get_default_session(), seq_list)
    #     return self.data_from_lstm(seqs)

    def get_log_probs_given_len(self, sample_list):
        sample_list_noend = [x[0:-1] for x in sample_list]
        logps = self.lstm.get_log_probs(tf.get_default_session(), sample_list_noend, max_batch_size=100)

        return logps

    def get_log_probs(self, sample_list):
        logps = self.get_log_probs_given_len(sample_list)

        lengths = [len(x) for x in sample_list]
        return logps + np.log(self.len_prob[lengths])

    # def get_log_probs_duel(self, sample_list):
    #     logpl = self.get_log_probs_given_len(sample_list)
    #
    #     lengths = [len(x) for x in sample_list]
    #     logp = logpl + np.log(self.len_prob[lengths])
    #     return logpl, logp

    def update(self, sample_list, scale, lr=None):
        # update lstm
        batch_size = self.sampler_config.max_batch_per_core
        for i in range(0, len(sample_list), batch_size):
            sample_list_noend = [x[0:-1] for x in sample_list[i: i+batch_size]]
            # sample_list_noend = sample_list
            self.lstm.update(tf.get_default_session(), sample_list_noend, scale[i: i+batch_size], learning_rate=lr)

    def save(self, fname):
        self.lstm.save(tf.get_default_session(), fname)

    def restore(self, fname):
        self.lstm.restore(tf.get_default_session(), fname)


class Ngram(Base):
    class Config(wb.Config):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.order = 2

        def __str__(self):
            return '{}gramGen'.format(self.order)

    def __init__(self, trf_config, data):
        super().__init__(trf_config)
        self.len_prob = trf_config.pi_0

        noise_config = noise.Config()
        noise_config.pack_size = trf_config.batch_size
        noise_config.min_len = trf_config.min_len
        noise_config.max_len = trf_config.max_len
        noise_config.beg_token = trf_config.beg_token
        noise_config.end_token = trf_config.end_token
        noise_config.vocab_size = trf_config.vocab_size
        noise_config.pi_true = self.len_prob

        ngramlm = ngram.Ngram(trf_config.sampler_config.order, trf_config.sampler_config.vocab_size)
        ngramlm.create_from_corpus(data_reduce_vocab(data.datas[0], trf_config.sampler_config.vocab_size))

        self.noise_sampler = noise.NoiseSamplerNgram(noise_config, data,
                                                     trf_config.sampler_config.order,
                                                     is_parallel=True,
                                                     input_ngram=ngramlm)

    def initialize(self):
        self.noise_sampler.start()

    def generate(self, num):
        sample_list = []

        while len(sample_list) < num:
            wlist, _ = self.noise_sampler.get(None)
            for w in wlist:
                if len(w) >= self.config.min_len:
                    sample_list.append(w)

        sample_list = sample_list[0:num]
        return sample_list

    def get_log_probs(self, sample_list):
        return self.noise_sampler.noise_logps(sample_list)

    def get_log_probs_duel(self, sample_list):
        logp = self.noise_sampler.noise_logps(sample_list)
        logpl = logp - np.log([self.noise_sampler.config.pi_true[len(x)] for x in sample_list])
        return logpl, logp

    def eval_nll(self, sample_list):
        logps = self.get_log_probs(sample_list)
        nll = -np.mean(logps)
        return nll




