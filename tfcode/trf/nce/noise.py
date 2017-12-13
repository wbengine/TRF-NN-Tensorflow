import tensorflow as tf
import numpy as np
import time
import json
import os
from collections import OrderedDict
from multiprocessing import Process, Manager, Queue, Value

from base import *
from lm import *


class Config(wb.Config):
    def __init__(self, trf_config):
        self.pack_size = trf_config.batch_size * trf_config.noise_factor  # the number of samples aftering calling get()
        self.min_len = trf_config.min_len
        self.max_len = trf_config.max_len
        self.beg_token = trf_config.beg_token
        self.end_token = trf_config.end_token
        self.vocab_size = trf_config.vocab_size
        self.pi_true = trf_config.pi_true


def create_noise_sampler(name, config, data, device='/gpu:0', logdir=None):
    """
    Args:
        name: the name of noise sampler
        config: noise.Config
        data: data
        device: device

    Returns:
        a instance of nosie sampler
    """
    if name.find('grampy') != -1:
        ngram = int(name.split('gram')[0])
        noise_sampler = NoiseSamplerNgramPy(config, data, ngram, logdir)
    elif name.find('gram') != -1:
        ngram = int(name.split('gram')[0])
        if ngram == 1:
            noise_sampler = NoiseSamplerUnigram(config, data)
        else:
            if name.find('train') != -1:
                noise_sampler = NoiseSamplerNgramTrain(config, data, ngram)
            elif name.find('nopi') != -1:
                noise_sampler = NoiseSamplerNgramNoPi(config, data, ngram)
            else:
                noise_sampler = NoiseSamplerNgram(config, data, ngram)
    elif name.find('lstm:') != -1:
        lstm_path = name.split(':')[-1]
        noise_sampler = NoiseSamplerLSTMEval(config, data, lstm_path, device=device)
    else:
        raise TypeError('undefined noise name =', name)

    return noise_sampler


class NoiseSampler(object):
    def __init__(self, config, data, name, is_parallel=False):
        self.config = config
        self.data = data
        self.name = name
        self.is_parallel = is_parallel

        print('[NoiseSampler] Type={} is_parallel={}'.format(name, is_parallel))

        if self.is_parallel:
            self.sample_queue = Queue(maxsize=10)
            self.sample_state = Value('i', 1)  # (i is on , 0 is off)
            self.sample_process = Process(target=self.sub_process,
                                          args=(self.sample_state,
                                                self.sample_queue,
                                                self.config.pack_size))

    def sub_process(self, state, sample_queue, num):
        while state.value == 1:
            if not sample_queue.full():
                seqs, logps = self.noise_generate(num)
                sample_queue.put((seqs, logps))
        print('[NoiseSampler] sub_process terminate')

    def get(self):
        # return self.config.noise_factor noise samples
        if self.is_parallel:
            return self.sample_queue.get()
        else:
            seqs = self.noise_generate(self.config.pack_size)
            return seqs, self.noise_logps(seqs)

    def start(self):
        # prepare for the sampling
        if self.is_parallel:
            print('[{}.{}] start sub process.'.format(__name__, self.__class__.__name__))
            self.sample_process.start()

    def release(self):
        # release the thread
        if self.is_parallel:
            self.sample_state.value = 0
            self.sample_process.join()

    def noise_generate(self, num):
        pass

    def noise_logps(self, seq_list):
        pass


class NoiseSamplerUnigram(NoiseSampler):
    def __init__(self, config, data, name='unigram'):
        self.unigram = data.get_unigram()
        # self.unigram[config.beg_token] = 1
        self.unigram[config.end_token] = 1
        self.unigram /= self.unigram.sum()

        super().__init__(config, data, name=name, is_parallel=True)

    def noise_generate(self, num):
        seqs = []
        while len(seqs) < num:
            rand_len = np.random.choice(self.config.max_len + 1, p=self.config.pi_true)
            # rand_len = length
            assert rand_len >= self.config.min_len
            assert rand_len <= self.config.max_len

            rand_s = np.random.choice(self.config.vocab_size, size=rand_len, p=self.unigram)
            rand_s[0] = self.config.beg_token
            # rand_s[-1] = self.config.end_token
            seqs.append(rand_s)

        logps = self.noise_logps(seqs)
        return seqs, logps

    def noise_logps(self, seq_list):
        logps = []
        for seq in seq_list:
            a = [self.unigram[i] for i in seq[1:]]

            if self.config.pi_true[len(seq)] == 0:
                raise TypeError('the input sequence is 0 len-prob, len={}, seq={}'.format(len(seq), seq))

            if np.any(np.array(a) == 0):
                raise TypeError('the prob is zero. probs={}, seq={}'.format(a, seq))
            
            logps.append(np.sum(np.log(a)) + np.log(self.config.pi_true[len(seq)]))

        return np.array(logps)


class NoiseSamplerNgram(NoiseSampler):
    def __init__(self, config, data, order, name=None):
        self.ngram = ngram.Ngram(order, data.get_vocab_size())
        self.ngram.create_from_corpus(data.datas[0])

        super().__init__(config, data,
                         name='%dgram' % order if name is None else name,
                         is_parallel=True)

    def noise_generate(self, num):
        seqs = []
        while len(seqs) < num:
            rand_len = np.random.choice(self.config.max_len + 1, p=self.config.pi_true)
            # rand_len = length
            assert rand_len >= self.config.min_len
            assert rand_len <= self.config.max_len

            rand_s = [self.config.beg_token]
            for _ in range(rand_len-1):
                p = self.ngram.get_prob(rand_s)
                w = p.sample()
                rand_s.append(w)

            # rand_s.append(self.config.end_token)
            seqs.append(rand_s)
        return seqs, self.noise_logps(seqs)

    def noise_logps(self, seq_list):
        logps = []
        for seq in seq_list:
            a = []
            for i in range(1, len(seq)):
                p = self.ngram.get_prob(seq[0:i])
                a.append(p[seq[i]])
            logps.append(np.sum(np.log(a)) + np.log(self.config.pi_true[len(seq)]))

        return np.array(logps)


class NoiseSamplerNgramNoPi(NoiseSamplerNgram):
    """
    Do not use the lenght distribution pi,
    make sure the last word is the end-token
    """
    def __init__(self, config, data, order, name=None):
        super().__init__(config, data, order, name='%dgram_nopi' % order)

    def noise_generate(self, num):
        max_len = self.config.max_len
        seqs = []
        while len(seqs) < num:
            rand_s = [self.config.beg_token]
            while len(rand_s) < max_len:
                p = self.ngram.get_prob(rand_s)
                w = p.sample()
                rand_s.append(w)
                if w == self.config.end_token:
                    break

            rand_s[-1] = self.config.end_token
            if len(rand_s) >= self.config.min_len:
                seqs.append(rand_s)
        return seqs, self.noise_logps(seqs)

    def noise_logps(self, seq_list):
        logps = []
        for seq in seq_list:
            a = []
            for i in range(1, len(seq)):
                p = self.ngram.get_prob(seq[0:i])
                a.append(p[seq[i]])
            logps.append(np.sum(np.log(a)))

        return np.array(logps)


class NoiseSamplerNgramPy(NoiseSampler):
    def __init__(self, config, data, order, logdir, name=None):
        ngram_config = pgramlm.Config(data)
        ngram_config.order = order
        self.ngram = pgramlm.Model(ngram_config, data, logdir)
        self.ngram.train()
        super().__init__(config, data,
                         name='%dgram_py' % order if name is None else name,
                         is_parallel=True)

    def noise_generate(self, num):
        seqs = []
        while len(seqs) < num:
            strings = self.ngram.gen(max_len=self.config.max_len+10)
            rand_s = []
            for w in strings[0:self.config.max_len]:
                if w == '</s>':
                    rand_s.append(self.config.end_token)
                elif w == '<unk>':
                    rand_s.append(self.data.get_unk_token())
                else:
                    rand_s.append(int(w))

            rand_s[-1] = self.config.end_token
            if len(rand_s) >= self.config.min_len:
                seqs.append(rand_s)
        return seqs, self.noise_logps(seqs)

    def noise_logps(self, seq_list):
        # logps = []
        # for seq in seq_list:
        #     logp = self.ngram.get_log_probs([seq])[0]
        #     logps.append(logp + np.log(self.config.pi_true[len(seq)]))

        return self.ngram.get_log_probs(seq_list)


class NoiseSamplerNgramTrain(NoiseSamplerNgram):
    def __init__(self, config, data, order):
        super().__init__(config, data, order, name='%gram' % order + '_train')

        self.choice_prob = 1 / (1 + self.config.noise_factor)
        self.data_prob = 1.0 / len(self.data.datas[0])

    def noise_generate(self, num):
        seqs = []
        probs = []
        while len(seqs) < num:
            if np.random.rand() <= self.choice_prob:
                # training set
                seqs.append(self.data.datas[0][np.random.randint(len(self.data.datas[0]))])
                probs.append(self.data_prob)
            else:
                rand_len = np.random.choice(self.config.max_len + 1, p=self.config.pi_true)
                # rand_len = length
                assert rand_len >= self.config.min_len
                assert rand_len <= self.config.max_len

                rand_s = [self.config.beg_token]
                for _ in range(rand_len-2):
                    p = self.ngram.get_prob(rand_s)
                    w = p.sample()
                    rand_s.append(w)

                rand_s.append(self.config.end_token)
                seqs.append(rand_s)
                probs.append(0)

        logps = []
        for data_prob, ngram_logp in zip(probs, super().noise_logps(seqs)):
            if data_prob == 0:
                logps.append(np.log(1-self.choice_prob) + ngram_logp)
            else:
                logps.append(
                    np.logaddexp(np.log(self.choice_prob) + np.log(data_prob),
                                 np.log(1-self.choice_prob) + ngram_logp)
                             )

        return seqs, np.array(logps)

    def noise_logps(self, seq_list):
        data_logp = np.log(1./self.data_prob)
        ngram_logp = super().noise_logps(seq_list)
        logps = np.logaddexp(np.log(self.choice_prob) + data_logp,
                             np.log(1-self.choice_prob) + ngram_logp)
        return logps


class NoiseSamplerLSTMEval(NoiseSampler):
    def __init__(self, config, data, lstm_path, device='/gpu:0'):
        if data.beg_token_str != data.end_token_str:
            raise ValueError('the data is not suitable for the Noise LSTM as '
                             'beg_token({}) != end_token({})'.format(data.beg_token_str, data.end_token_str))
        self.lstm = lstmlm.LM.load(lstm_path, device)
        self.lstm_path = lstm_path
        self.device = device
        super().__init__(config, data, name='lstm_eval', is_parallel=False)

    def start(self):
        super().start()
        print('[{}.{}] restore the lstm'.format(__name__, self.__class__.__name__))
        self.lstm.restore(tf.get_default_session(), self.lstm_path)

    def noise_generate(self, num):
        rand_lens = np.random.choice(self.config.max_len + 1, size=num, p=self.config.pi_true)
        max_rand_len = np.max(rand_lens)
        rand_seqs, _ = self.lstm.simulate(tf.get_default_session(),
                                          self.config.beg_token * np.ones((num, 1), dtype='int32'),
                                          int(max_rand_len-1), initial_state=True)
        rand_seqs[np.arange(num), rand_lens-1] = self.config.end_token

        if np.max(rand_seqs) >= self.config.vocab_size:
            print('[NoiseSamplerLSTMEval] warnning: generate illegal word {}'.format(np.max(rand_seqs)))
            print(rand_seqs)
            print(rand_lens)
            rand_seqs = np.minimum(rand_seqs, self.config.vocab_size-1)

        seqs = reader.extract_data_from_trf(rand_seqs, rand_lens)
        return seqs

    def noise_logps(self, seq_list):
        inputs, lengths = reader.produce_data_to_trf(seq_list)
        # return self.lstm.conditional(tf.get_default_session(), inputs, 1, lengths, initial_state=True)

        len_logps = np.log([self.config.pi_true[i] for i in lengths])
        seq_logps = self.lstm.conditional(tf.get_default_session(), inputs, 1, lengths-1, initial_state=True)
        logps = len_logps + seq_logps
        return np.array(logps)


class NoiseSamplerLSTMGene(NoiseSampler):
    def __init__(self, config, data, lstm_path, device='/gpu:0'):
        self.lstm_path = lstm_path
        self.device = device
        super().__init__(config, data, name='lstm_gene', is_parallel=True)

    def sub_process(self, state, q, batch_num):

        with tf.Graph().as_default():
            log_label = '[NoiseSamplerLSTM-subprocess]'
            print(log_label, 'create lstm')
            lm = lstmlm.LM.load(self.lstm_path, self.device)

            print(log_label, 'create session')
            session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_config.gpu_options.allow_growth = True
            with tf.Session(config=session_config) as session:
                lm.restore(session, self.lstm_path)

                while state.value == 1:
                    if not q.full():
                        seqs, logps = self.sub_process_noise_generate(session, lm, batch_num)
                        q.put((seqs, logps))

        print(log_label, 'sub_process terminate')

    def sub_process_noise_generate(self, session, lm, num):
        rand_lens = np.random.choice(self.config.max_len + 1, size=num, p=self.config.pi_true)
        max_rand_len = np.max(rand_lens)
        rand_seqs, _ = lm.simulate(session,
                                   self.config.beg_token * np.ones((num, 1), dtype='int32'),
                                   int(max_rand_len-1), initial_state=True)
        rand_seqs[np.arange(num), rand_lens-1] = self.config.end_token
        seqs = reader.extract_data_from_trf(rand_seqs, rand_lens)

        len_logps = np.log([self.config.pi_true[i] for i in rand_lens])
        seq_logps = lm.conditional(session, rand_seqs, 1, rand_lens-1, initial_state=True)
        logps = len_logps + seq_logps
        return seqs, logps


class NoiseSamplerLSTM(NoiseSampler):
    def __init__(self, config, data, lstm_path, device='/gpu:0'):
        super().__init__(config, data, name='lstm', is_parallel=False)
        self.eval = NoiseSamplerLSTMEval(config, data, lstm_path, device)
        self.gene = NoiseSamplerLSTMGene(config, data, lstm_path, device)

    def start(self):
        self.eval.start()
        self.gene.start()

    def release(self):
        self.gene.release()
        self.eval.release()

    def get(self):
        return self.gene.get()

    def noise_logps(self, seq_list):
        return self.eval.noise_logps(seq_list)

    def noise_generate(self, num):
        return self.eval.noise_generate(num)
