import os
import tensorflow as tf
import json

from base import *
from trf.common import feat2 as feat
from . import densefeat, alg


def logab_int(a, b):
    """compute the n in function a**n = b, with a, b are both integer"""
    n = 0
    while b != 1:
        n += 1
        b = b//a
    return n


class TagConfig(wb.Config):
    def __init__(self, data):
        """
        Args:
            data: seq.Data()
        """
        self.feat_dict = {'c[1:2]': 0}
        self.L2_reg = 0
        self.tag_size = data.get_tag_size()

    def __str__(self):
        max_order = feat.Feats(self.feat_dict).get_order()
        return 't{}g'.format(max_order)


class TagBigramConfig(wb.Config):
    def __init__(self, data):
        """
        Args:
            data: seq.Data()
        """
        self.L2_reg = 0
        self.tag_size = data.get_tag_size()

    def __str__(self):
        return 'tagbigram'


class TagPhi(object):
    def __init__(self, config, data_seq_list, opt_method):
        self.config = config
        self.data_seq_list = data_seq_list
        self.opt_method = opt_method

        # tag features
        # if wb.is_linux():
        #     self.feats = feat.FastFeats(self.config.feat_dict)
        # else:
        self.feats = feat.Feats(self.config.feat_dict)

        # update
        self.update_op = None

        # save the trans_matrix
        self.trans_matrix = np.zeros([self.config.tag_size ** (self.get_order()-1)] * 2)
        self.trans_matrix_tail = np.zeros_like(self.trans_matrix)  # the trans_matrix at the last position
        self.need_update_trans_matrix = True  # if true, then recompute the trans_matrix

        # used to compute the get_exp()
        self.tag_map_ids = None
        self.tag_map_ids_extra = None

    def get_param_num(self):
        return self.feats.num

    def get_order(self):
        return self.feats.get_order()

    def get_value(self, seq_list, depend_on=None):
        w = self.feats.seq_list_weight(seq_list, depend_on)
        return np.array(w)

    def set_params(self, value=None):
        if value is None:
            self.feats.values = np.random.uniform(-0.1, 0.1, size=self.get_param_num())
        else:
            self.feats.values = np.ones(self.get_param_num()) * value
        self.need_update_trans_matrix = True

    def get_propose_logps(self, seq_list, tag_pos):
        if isinstance(tag_pos, int):
            tag_pos = [tag_pos] * len(seq_list)

        logps = []
        for s, pos in zip(seq_list, tag_pos):
            temp_seqs = seq.seq_list_enumerate_tag([s], self.config.tag_size, pos)
            m = self.get_value(temp_seqs, depend_on=(1, pos))
            logps.append(m)

        return np.array(logps)

    def get_gradient(self, data_list, data_scalar, sample_list, sample_scalar):
        exp_d = self.feats.seq_list_count(data_list, data_scalar)
        exp_s = self.feats.seq_list_count(sample_list, sample_scalar)
        return exp_d - exp_s - self.config.L2_reg * self.feats.values

    def get_exp(self, data_list, data_scalar, logps_list=None):
        if logps_list is None:
            # using the given tags
            return self.feats.seq_list_count(data_list, data_scalar)

        tag_map_ids, tag_map_ids_extra = self.get_tag_map_ids()

        exp_s = np.zeros_like(self.feats.values)
        for scalar, logps in zip(data_scalar, logps_list):

            order = logab_int(self.config.tag_size, logps.shape[1])
            assert order == self.get_order()

            logps_sum = logsumexp(logps, axis=0)  # sum all position
            for ids, p_for_tag in zip(tag_map_ids, logps_sum):
                for k in ids:
                    exp_s[k] += np.exp(p_for_tag) * scalar

            for ids, p_for_tag in zip(tag_map_ids_extra, logps[-1]):
                for k in ids:
                    exp_s[k] += np.exp(p_for_tag) * scalar

        return exp_s

    # def get_gradient_fb(self, data_list, data_scalar,  sample_list, sample_scalar, logps_list):
    #     exp_d = self.feats.seq_list_count(data_list, data_scalar)
    #     exp_s = self.get_exp(sample_list, sample_scalar, logps_list)
    #     return exp_d - exp_s - self.config.L2_reg * self.feats.values

    def update(self, data_list, data_scalars, sample_list, sample_scalars, learning_rate=1.0,
               data_fp_logps_list=None,
               sample_fp_logps_list=None):
        exp_d = self.get_exp(data_list, data_scalars, data_fp_logps_list)
        exp_s = self.get_exp(sample_list, sample_scalars, sample_fp_logps_list)
        g = exp_d - exp_s - self.config.L2_reg * self.feats.values

        if self.update_op is None:
            raise TypeError('[{}.{}] update: the update_op=None, please run self.initialize() first!'.format(
                __name__, self.__class__.__name__
            ))

        d = self.update_op.update(-g, learning_rate)
        self.feats.values += d
        self.need_update_trans_matrix = True

    def initialize(self):
        if self.feats.num == 0:
            self.feats.load_from_seqs(self.data_seq_list)
        else:
            print('[{}.{}] Features exist. Don\'t reload features'.format(__name__, self.__class__.__name__))

        self.update_op = wb.ArrayUpdate(self.get_param_num(), {'name': self.opt_method})

    def save(self, fname):
        with open(fname + '.tag.feat', 'wt') as f:
            self.feats.save(f)

    def restore(self, fname):
        with open(fname + '.tag.feat', 'rt') as f:
            self.feats.restore(f)

        # update op
        self.update_op = wb.ArrayUpdate(self.get_param_num(), {'name': self.opt_method})

    def get_tag_map_ids(self):
        if self.tag_map_ids is not None and self.tag_map_ids_extra is not None:
            return self.tag_map_ids, self.tag_map_ids_extra

        order = self.get_order()
        self.tag_map_ids = [0] * (self.config.tag_size ** order)
        self.tag_map_ids_extra = [0] * (self.config.tag_size ** order)

        s = seq.Seq(order)
        for tags in sp.VecIter(order, self.config.tag_size):
            s.x[1] = tags
            ids = self.feats.ngram_find(s)
            self.tag_map_ids[sp.map_list(tags, self.config.tag_size)] = ids

            ids = []
            for j in range(1, order):
                ids += self.feats.ngram_find(s[j:])
            self.tag_map_ids_extra[sp.map_list(tags, self.config.tag_size)] = ids

        return self.tag_map_ids, self.tag_map_ids_extra

    def get_trans_matrix(self):
        """compute the trans-matrix for forward-backward algorithms,
            return the log probs
        """
        if self.need_update_trans_matrix:
            # recompute the trans_matrix
            order = self.get_order()
            self.trans_matrix = np.ones([self.config.tag_size ** (order-1)] * 2) * alg.LOG_ZERO  # log-zero-prob
            self.trans_matrix_tail = np.array(self.trans_matrix)  # the trans_matrix at the last position

            s = seq.Seq(order)
            for tags in sp.VecIter(order, self.config.tag_size):
                i = sp.map_list(tags[0:-1], self.config.tag_size)
                j = sp.map_list(tags[1:], self.config.tag_size)
                s.x[1] = tags
                v = self.feats.ngram_weight(s)
                self.trans_matrix[i, j] = v

                # trans_matrix for the last position
                for k in range(1, len(s)):
                    v += self.feats.ngram_weight(s[k:])
                self.trans_matrix_tail[i, j] = v

            # compute e^phi
            # self.trans_matrix = np.exp(self.trans_matrix)
            # self.trans_matrix_tail = np.exp(self.trans_matrix_tail)

            self.need_update_trans_matrix = False
        # return the log-probs
        return self.trans_matrix, self.trans_matrix_tail


class TagBigram(object):
    def __init__(self, config, data_seq_list, opt_method):
        self.config = config
        self.data_seq_list = data_seq_list
        self.opt_method = opt_method

        # tag features
        self.edge_mat = np.zeros([self.config.tag_size, self.config.tag_size])

        # update
        self.update_op = wb.ArrayUpdate(self.edge_mat, {'name': opt_method, 'max_norm': 10})

    def initialize(self):
        pass

    def get_param_num(self):
        return np.size(self.edge_mat)

    def get_order(self):
        return 2

    def get_value(self, seq_list, depend_on=None):
        values = []

        if depend_on is None:
            for t in seq.get_h(seq_list):
                a = self.edge_mat[t[0:-1], t[1:]]
                values.append(np.sum(a))
        else:
            for t in seq.get_h(seq_list):
                n = depend_on[-1]  # position
                v = 0
                if n-1 >= 0:
                    v += self.edge_mat[t[n-1], t[n]]
                if n+1 <= len(t) - 1:
                    v += self.edge_mat[t[n], t[n+1]]
                values.append(v)
        return np.array(values)

    def set_params(self, value=None):
        if value is None:
            self.edge_mat = np.random.uniform(-0.1, 0.1, size=self.edge_mat.shape)
        else:
            self.edge_mat = value

    def get_propose_logps(self, seq_list, tag_pos):
        if isinstance(tag_pos, int):
            tag_pos = [tag_pos] * len(seq_list)

        logps = []
        for s, pos in zip(seq_list, tag_pos):
            temp_seqs = seq.seq_list_enumerate_tag([s], self.config.tag_size, pos)
            m = self.get_value(temp_seqs, depend_on=(1, pos))
            logps.append(m)

        return np.array(logps)

    def get_exp(self, data_list, data_scalar, logps_list=None):
        exp_s = np.zeros_like(self.edge_mat)

        if logps_list is None:
            for scalar, seq in zip(data_scalar, data_list):
                t = seq.x[1]
                for i in range(len(t)-1):
                    exp_s[t[i], t[i+1]] += scalar
        else:
            for scalar, logps in zip(data_scalar, logps_list):

                order = logab_int(self.config.tag_size, logps.shape[1])
                assert order == self.get_order()

                logps_sum = logsumexp(logps, axis=0)  # sum all position
                exp_s += scalar * logps_sum.reshape([self.config.tag_size, self.config.tag_size])

        return exp_s

    def update(self, data_list, data_scalars, sample_list, sample_scalars, learning_rate=1.0,
               data_fp_logps_list=None,
               sample_fp_logps_list=None):
        exp_d = self.get_exp(data_list, data_scalars, data_fp_logps_list)
        exp_s = self.get_exp(sample_list, sample_scalars, sample_fp_logps_list)
        g = exp_d - exp_s - self.config.L2_reg * self.edge_mat

        d = self.update_op.update(-g, learning_rate)
        self.edge_mat += d

    def save(self, fname):
        with open(fname + '.tag.mat', 'wt') as f:
            json.dump({'edge': self.edge_mat.tolist()}, f, indent=4)

    def restore(self, fname):
        with open(fname + '.tag.mat', 'rt') as f:
            a = json.load(f)
            self.edge_mat = np.array(a['edge'])

        # update op
        self.update_op = wb.ArrayUpdate(self.edge_mat, {'name': self.opt_method})

    def get_trans_matrix(self):
        """compute the trans-matrix for forward-backward algorithms,
            return the log probs
        """
        # return the log-probs
        return self.edge_mat, self.edge_mat











