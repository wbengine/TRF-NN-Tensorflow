########################################
# This is a ngram LM wrapping using the srilm-python toolkits
########################################

import sys
sys.path.insert(0, '../../tools/srilm/srilm-python')

import os
import array
import numpy as np

from base import *

import srilm.vocab
import srilm.stats
import srilm.ngram
import srilm.discount


class Config(wb.Config):
    def __init__(self, data):
        self.vocab_size = data.get_vocab_size()
        self.order = 3
        self.discount_method = 'kneser-ney'
        self.is_interpolate = True
        self.cutoff = [0]


class Model(object):
    def __init__(self, config, data, logdir=None):
        self.config = config
        self.data = data
        self.logdir = logdir

        # create vocab
        self.vocab = srilm.vocab.Vocab()
        for i, w in enumerate(data.word_list):
            self.vocab.add(str(i))

        self.word_idxs = []
        for w, idx in self.vocab:
            self.word_idxs.append(idx)
        self.word_idxs.sort()
        if self.logdir is not None:
            self.vocab.write(os.path.join(self.logdir, 'vocab'))

        # model
        self.lm = None

    def ngram_logp(self, word_index_ngram):
        word_index_ngram = word_index_ngram[-self.config.order:]
        log10p = self.lm.prob(word_index_ngram[-1],
                              array.array('i', reversed(word_index_ngram[0:-1])))
        # as the logp in SRILM is base log10,
        # transform to log
        return log10p / np.log10(np.exp(1))

    def get_log_probs(self, seq_list):
        logps = []
        for s in seq_list:
            # s_str = [str(i) for i in s]
            # s_str.insert(0, '<s>')
            # s_str.append('</s>')
            idx = list(self.vocab.index([str(i) for i in s]))
            idx.insert(0, self.vocab.bos)
            idx.append(self.vocab.eos)
            logp = 0
            for i in range(1, len(idx)):
                logp += self.ngram_logp(idx[0:i+1])

            # as the logp in SRILM is base log10,
            # transform to log
            logps.append(logp)
        return np.array(logps)

    def eval(self, seq_list):
        logps = self.get_log_probs(seq_list)
        nll = -np.mean(logps)
        words = np.sum([len(x) for x in seq_list])
        words += len(seq_list)  # add end-tokens
        print(words)
        ppl = np.exp(-np.sum(logps) / words)
        return nll, ppl

    def cond(self, seq, beg_pos=0, end_pos=None):
        """
        compute the log p(s[beg_pos: end_pos] | s[0: beg_pos])
        """
        if end_pos is None:
            end_pos = len(seq)

        idxs = list(self.vocab.index([str(i) for i in seq]))
        idxs.insert(0, self.vocab.bos)
        idxs.append(self.vocab.eos)

        logp = 0
        for i in range(beg_pos+1, end_pos+1):
            logp += self.ngram_logp(idxs[0:i+1])
        return logp

    def gen(self, fixed_len=None, max_len=100):
        if fixed_len is None:
            return self.lm.rand_gen(max_len)
        else:
            s = self.lm.rand_gen(fixed_len+1)
            while len(s) < fixed_len:
                s += self.lm.rand_gen(fixed_len + 1 - len(s))
            return s
            # idx = [self.vocab.bos] * (fixed_len + 1)
            # for i in range(1, fixed_len + 1):
            #     rand_uniform = np.random.uniform()
            #
            #     acc_p = 0
            #     for widx in self.word_idxs:
            #         if widx == self.vocab.bos:
            #             continue
            #         idx[i] = widx
            #         acc_p += np.exp(self.ngram_logp(idx[0:i+1]))
            #         if acc_p >= rand_uniform:
            #             break
            # return self.vocab.string(idx[1:])

    def train(self, write_to_res=None):

        assert(self.logdir is not None)

        data = self.data

        ngram_states = []
        id_files = [os.path.join(self.logdir, fname + '.id') for fname in ['train', 'valid', 'test']]
        for seq_list, fname in zip(data.datas, id_files):
            st = srilm.stats.Stats(self.vocab, self.config.order)
            data.write_data(seq_list, fname)
            st.count_file(fname)
            ngram_states.append(st)

        self.lm = srilm.ngram.Lm(self.vocab, self.config.order)
        for i in range(self.config.order):
            self.lm.set_discount(i + 1,
                                 srilm.discount.Discount(
                                     method=self.config.discount_method,
                                     interpolate=self.config.is_interpolate,
                                     min_count=self.config.cutoff[min(i, len(self.config.cutoff)-1)])
                                 )
        self.lm.train(ngram_states[0])

        # write to res
        if write_to_res is not None:
            res_file = write_to_res[0]
            res_name = write_to_res[1]

            # print(np.sum(self.get_log_probs(data.datas[0])))
            # print(self.eval(data.datas[0]))
            # print(self.lm.test_text_file(id_files[0]))

            ppls = [self.lm.test_text_file(fname)[-1] for fname in id_files]
            print('ppls =', ppls)

            res = wb.FRes(res_file)
            res.AddPPL(res_name, ppls, id_files)










