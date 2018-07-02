import numpy as np
import tensorflow as tf

from scipy.misc import logsumexp
from base import *
from semi import alg as alg_tf

LOG_ZERO = -1e16


class ForwardBackward(object):
    """

     alpha_0       alpha_1       alpha_2       alpha_3  ...  alpha_l-1
    --------> x_0 --------> x_1 --------> x_2 --------> ... --------> x_l-1 -------->
    <-------- x_0 <-------- x_1 <-------- x_2 <-------- ... <-------- x_l-1 <--------
                   beta_0        beta_1        beta_2         beta_l-2       beta_l-1

    """
    def __init__(self, trans_mat, trans_mat_last, emiss_mat, beg_idxs=None, end_idxs=None):
        self.trans_mat = trans_mat  # logprob
        self.trans_mat_last = trans_mat_last # logprob
        self.emiss_mat = emiss_mat  # logprob
        self.node_size = trans_mat.shape[0]
        self.node_num = emiss_mat.shape[0]
        self.beg_idxs = np.array(beg_idxs) if beg_idxs is not None else np.arange(0, self.node_size)
        self.end_idxs = np.array(end_idxs) if end_idxs is not None else np.arange(0, self.node_size)

        assert emiss_mat.shape[1] == self.node_size
        assert trans_mat.shape[0] == trans_mat.shape[1]

        self.alpha = None
        self.beta = None

    def get_mask(self):
        beg_mask = np.zeros(self.node_size, dtype='bool')
        end_mask = np.zeros(self.node_size, dtype='bool')
        beg_mask[self.beg_idxs] = True
        end_mask[self.end_idxs] = True
        return beg_mask, end_mask

    def forward(self):
        alpha = np.zeros((self.node_num, self.node_size))

        beg_mask, end_mask = self.get_mask()

        alpha[0] = np.where(beg_mask, self.emiss_mat[0], LOG_ZERO)
        for i in range(self.node_num-1):
            if i == self.node_num-2:
                a = self.trans_mat_last
            else:
                a = self.trans_mat

            alpha[i+1] = sp.logsumexp(np.reshape(alpha[i], [-1, 1]) + a, axis=0) + self.emiss_mat[i+1]

        alpha[-1] = np.where(end_mask, alpha[-1], LOG_ZERO)

        return alpha  # the log-alpha

    def backward(self):
        beta = np.zeros((self.node_num, self.node_size))

        beg_mask, end_mask = self.get_mask()

        beta[-1] = np.where(end_mask, self.emiss_mat[-1], LOG_ZERO)
        for i in reversed(range(self.node_num-1)):
            if i == self.node_num-2:
                a = self.trans_mat_last
            else:
                a = self.trans_mat

            beta[i] = sp.logsumexp(a + np.reshape(beta[i+1], [1, -1]), axis=-1) + self.emiss_mat[i]

        beta[0] = np.where(beg_mask, beta[0], LOG_ZERO)

        return beta  # log-beta

    def decode(self):

        beg_mask, end_mask = self.get_mask()
        optimal_perv = np.ones([self.node_num, self.node_size], dtype='int32') * (-1)

        m = np.where(beg_mask, self.emiss_mat[0], LOG_ZERO)
        for i in range(self.node_num-1):  # forward max_time-1 times
            if i == self.node_num-2:
                a = self.trans_mat_last  # the trans_matrix
            else:
                a = self.trans_mat

            weight = np.reshape(m, [-1, 1]) + a + self.emiss_mat[i+1]
            opt_i = np.argmax(weight, axis=0)

            optimal_perv[i+1] = opt_i
            m = weight[opt_i, np.arange(0, self.node_size)]

        m = np.where(end_mask, m, LOG_ZERO)
        opt_i = np.argmax(m)
        opt_w = m[opt_i]

        opt_tags = [0] * self.node_num
        for t in reversed(range(0, self.node_num)):
            opt_tags[t] = opt_i
            opt_i = optimal_perv[t][opt_i]

        return opt_tags, opt_w

    def logsum(self):
        """return the log-summation"""
        # probs = self.probs()
        # print(np.log(np.sum(np.sum(probs, axis=-1), axis=-1)))

        if self.alpha is None:
            self.alpha = self.forward()
        return sp.logsumexp(self.alpha[-1])

    def logp_at_position(self, i):
        if i == self.node_num-2:
            a = self.trans_mat_last
        else:
            a = self.trans_mat

        p = np.reshape(self.alpha[i], [-1, 1]) + a + np.reshape(self.beta[i+1], [1, -1])
        return p

    def logps(self):
        """
        compute the marignial 2gram log-prob at each position
        Returns:
            a array of shape (self.node_num-1, self.node_size, self.node_size)
        """
        if self.alpha is None:
            self.alpha = self.forward()
        if self.beta is None:
            self.beta = self.backward()

        logps = np.zeros((self.node_num-1, self.node_size, self.node_size))
        for i in range(self.node_num - 1):
            logps[i] = self.logp_at_position(i)

        return logps


def logab_int(a, b):
    """compute the n in function a**n = b, with a, b are both integer"""
    n = 0
    while b != 1:
        n += 1
        b = b//a
    return n


def marginal_logps_list(logps_list, vocab_size, to_order):
    return [marginal_logps(logps, vocab_size, to_order) for logps in logps_list]


def marginal_logps(logps, vocab_size, to_order):
    """
    the output of FB.logps is the logp of the max-order.
    Some times we need the mariginal logps to a lower order.
    This function is to do this
    Args:
        logps: the logps of one sequences at each position, of a larger order,
                    np.array() of shape [seq_len, vocab_size**order],
        vocab_size: the vocab_size of this problems
        to_order:   the needed order, should be less than the original order

    Returns:
        a logps list of order 'to_order'
    """

    order = logab_int(vocab_size, logps.shape[1])

    if to_order > order:
        raise TypeError('[%s] marginal_logps_list: the need order={} is larger than the given order={}'.format(
            __name__, to_order, order))

    if to_order == order:
        return logps

    final_logps = []
    ####################################################################
    # This operation is depending on the function in sp.Map_list() !!!!
    src_logps = np.reshape(logps, [-1] + [vocab_size] * order)
    for p in src_logps:
        p = logsumexp(p, axis=tuple(range(to_order, order)))
        final_logps.append(np.reshape(p, [-1]))

    for i in range(1, order-to_order+1):
        axis = list(range(0, i)) + list(range(i+to_order, order))
        p = logsumexp(src_logps[-1], axis=tuple(axis))
        final_logps.append(np.reshape(p, [-1]))

    return np.array(final_logps)


def logps_list_package(logps_list, max_len=None):
    batch_size = len(logps_list)
    lengths = [p.shape[0] for p in logps_list]
    dims = [p.shape[1] for p in logps_list]

    if max_len is None:
        max_len = np.max(lengths)

    assert max_len >= np.max(lengths)
    assert np.all(np.array(dims) == dims[0])

    logps_array = np.ones([batch_size, max_len, dims[0]]) * LOG_ZERO
    for i, p in enumerate(logps_list):
        logps_array[i][0: p.shape[0]] = p

    return logps_array


def logps_list_unfold(logps_array, lengths):
    logps_list = []
    for logps, n in zip(logps_array, lengths):
        logps_list.append(logps[0:n])
    return logps_list


# class ForwardBackward_tf(ForwardBackward):
#     def __init__(self, trans_mat, trans_mat_last, emiss_mat, beg_idxs=None, end_idxs=None):
#         assert np.all(trans_mat == trans_mat_last)

def decode(trans_mats, beg_idxs, end_idxs):
    """
    Args:
        trans_mats: np.array, [length-1, node_size, node_size]
        beg_idxs: list
        end_idxs: list

    Returns:

    """
    node_num = trans_mats.shape[0] + 1
    node_size = trans_mats.shape[1]
    # mask
    beg_mask = np.zeros(node_size, dtype='bool')
    end_mask = np.zeros(node_size, dtype='bool')
    beg_mask[beg_idxs] = True
    end_mask[end_idxs] = True

    optimal_perv = np.ones([node_num, node_size], dtype='int32') * (-1)

    m = np.where(beg_mask, 0, LOG_ZERO)
    for i in range(node_num - 1):  # forward max_time-1 times
        a = trans_mats[i]

        weight = np.reshape(m, [-1, 1]) + a
        opt_i = np.argmax(weight, axis=0)

        optimal_perv[i + 1] = opt_i
        m = weight[opt_i, np.arange(0, node_size)]

    m = np.where(end_mask, m, LOG_ZERO)
    opt_i = np.argmax(m)
    opt_w = m[opt_i]

    opt_tags = [0] * node_num
    for t in reversed(range(0, node_num)):
        opt_tags[t] = opt_i
        opt_i = optimal_perv[t][opt_i]

    return opt_tags, opt_w
