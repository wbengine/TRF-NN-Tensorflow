import tensorflow as tf
import numpy as np

LOG_ZERO = -1e16


# Forward-backward algorithm
#
#
#  alpha_0       alpha_1       alpha_2       alpha_3  ...  alpha_l-1
# --------> x_0 --------> x_1 --------> x_2 --------> ... --------> x_l-1 -------->
# <-------- x_0 <-------- x_1 <-------- x_2 <-------- ... <-------- x_l-1 <--------
#                beta_0        beta_1        beta_2         beta_l-2       beta_l-1


def get_mask(ids, batch_size, node_size):
    m = tf.scatter_nd(indices=tf.reshape(ids, [-1, 1]),
                      updates=tf.ones_like(ids),
                      shape=[node_size])  # [node_size]
    m = tf.tile(tf.reshape(m, [1, -1]), multiples=[batch_size, 1])  # [batch_size, node_size]
    return tf.cast(m, dtype=tf.bool)


def forward_tf(trans_logps, emiss_logps, lengths, beg_ids, end_ids):
    """
    perform forward.
    'node_size' denotes the tag size in tagging task.
    Args:
        trans_logps: Tensor, [node_size, node_size]
        emiss_logps: Tensor, [batch_size, seq_max_len, node_size]
        lengths: Tensor, [batch_size], the length of each sequences
        beg_ids: a list of begin ids
        end_ids: a list of end ids

    Returns:
        alphas, Tensor, [batch_size, seq_max_len+1, node_size]
    """
    node_size = tf.shape(trans_logps)[0]
    batch_size = tf.shape(emiss_logps)[0]
    max_len = tf.shape(emiss_logps)[1]
    beg_mask = get_mask(beg_ids, batch_size, node_size)
    end_mask = get_mask(end_ids, batch_size, node_size)

    trans_logps_expand = tf.expand_dims(trans_logps, axis=0)  # [1, node_size, node_size]

    # alpha, [batch_size, node_size]
    alpha_init = tf.where(beg_mask, emiss_logps[:, 0], tf.ones_like(emiss_logps[:, 0]) * LOG_ZERO)

    def fn(curr_alpha, i):
        # curr_alpha: [batch_size, node_size]
        # i: int
        a = tf.expand_dims(curr_alpha, axis=-1)  # [batch_size, node_size, 1]
        a = tf.reduce_logsumexp(a + trans_logps_expand, axis=1) + emiss_logps[:, i+1]  # [batch_size, node_size]
        return a

    alphas = tf.scan(fn, tf.range(0, max_len - 1), initializer=alpha_init)  # [max_len, batch_size, node_size]
    alphas = tf.transpose(alphas, [1, 0, 2])  # [batch_size, max_len, node_size]

    # add alpha_init
    alphas = tf.concat([tf.expand_dims(alpha_init, axis=1), alphas], axis=1)

    # end mask
    end_mask = tf.scatter_nd(indices=tf.stack([tf.range(0, batch_size), lengths-1], axis=1),
                             updates=1 - tf.cast(end_mask, tf.int32),
                             shape=[batch_size, max_len, node_size])  # [batch_size, max_len, node_size]
    alphas = tf.where(tf.cast(end_mask, tf.bool), tf.ones_like(alphas) * LOG_ZERO, alphas)

    return alphas


def forward_tf2(trans_logps, lengths, beg_ids, end_ids):
    """
        perform forward.
        'node_size' denotes the tag size in tagging task.
        Args:
            trans_logps: Tensor, [batch_size, max_len, node_size, node_size]
            lengths: Tensor, [batch_size], the length of each sequences
            beg_ids: a list of begin ids
            end_ids: a list of end ids

        Returns:
            alphas, Tensor, [batch_size, seq_max_len+1, node_size]
    """
    node_size = tf.shape(trans_logps)[-1]
    batch_size = tf.shape(trans_logps)[0]
    max_len = tf.shape(trans_logps)[1] + 1
    beg_mask = get_mask(beg_ids, batch_size, node_size)
    end_mask = get_mask(end_ids, batch_size, node_size)

    # alpha, [batch_size, node_size]
    alpha_init = tf.where(beg_mask,
                          tf.zeros([batch_size, node_size], dtype=tf.float32),
                          tf.ones([batch_size, node_size], dtype=tf.float32) * LOG_ZERO)

    def fn(curr_alpha, i):
        # curr_alpha: [batch_size, node_size]
        # i: int
        a = tf.expand_dims(curr_alpha, axis=-1)  # [batch_size, node_size, 1]
        a = tf.reduce_logsumexp(a + trans_logps[:, i], axis=1)  # [batch_size, node_size]
        return a

    alphas = tf.scan(fn, tf.range(0, max_len - 1), initializer=alpha_init)  # [max_len, batch_size, node_size]
    alphas = tf.transpose(alphas, [1, 0, 2])  # [batch_size, max_len, node_size]

    # add alpha_init
    alphas = tf.concat([tf.expand_dims(alpha_init, axis=1), alphas], axis=1)

    # end mask
    end_mask = tf.scatter_nd(indices=tf.stack([tf.range(0, batch_size), lengths - 1], axis=1),
                             updates=1 - tf.cast(end_mask, tf.int32),
                             shape=[batch_size, max_len, node_size])  # [batch_size, max_len, node_size]
    alphas = tf.where(tf.cast(end_mask, tf.bool), tf.ones_like(alphas) * LOG_ZERO, alphas)

    return alphas


def backward_tf(trans_logps, emiss_logps, lengths, beg_ids, end_ids):
    """
    perform forward.
    'node_size' denotes the tag size in tagging task.
    Args:
        trans_logps: Tensor, [node_size, node_size]
        emiss_logps: Tensor, [batch_size, seq_max_len, node_size]
        lengths: Tensor, [batch_size], the length of each sequences
        beg_ids: a list of begin ids
        end_ids: a list of end ids

    Returns:
        alphas, Tensor, [batch_size, seq_max_len+1, node_size]
    """
    node_size = tf.shape(trans_logps)[0]
    batch_size = tf.shape(emiss_logps)[0]
    max_len = tf.shape(emiss_logps)[1]
    beg_mask = get_mask(beg_ids, batch_size, node_size)
    end_mask = get_mask(end_ids, batch_size, node_size)

    trans_logps_expand = tf.expand_dims(trans_logps, axis=0)  # [1, node_size, node_size]
    emiss_logps_reverse = tf.reverse_sequence(emiss_logps, lengths, seq_dim=1)

    beta_init = tf.where(end_mask, emiss_logps_reverse[:, 0], tf.ones_like(emiss_logps_reverse[:, 0]) * LOG_ZERO)

    def fn(curr_beta, i):
        # curr_alpha: [batch_size, node_size]
        # i: int
        b = tf.expand_dims(curr_beta, axis=1)  # [batch_size, 1, node_size]
        #  [batch_size, node_size]
        b = tf.reduce_logsumexp(trans_logps_expand + b, axis=-1) + emiss_logps_reverse[:, i+1]
        return b

    betas = tf.scan(fn, tf.range(0, max_len-1), initializer=beta_init)  # [max_len, batch_size, node_size]
    betas = tf.transpose(betas, [1, 0, 2])  # [batch_size, max_len, node_size]
    # add beta_init
    betas = tf.concat([tf.expand_dims(beta_init, axis=1), betas], axis=1)
    # reverse
    betas = tf.reverse_sequence(betas, lengths, seq_dim=1)

    # beg mask
    beg_mask = tf.where(beg_mask, betas[:, 0], tf.ones_like(betas[:, 0]) * LOG_ZERO)
    betas = tf.concat([tf.expand_dims(beg_mask, axis=1), betas[:, 1:]], axis=1)

    return betas


def get_logps(trans_logps, alphas, betas):
    """
    compute the marignial 2gram log-prob at each position
    Args:
        trans_logps: Tensor, [node_size, node_size]
        alphas: tensor, [batch_size, max_len, node_size]
        betas:  tensor, [batch_size, max_len, node_size]

    Returns:

    """
    node_size = tf.shape(trans_logps)[0]

    a = tf.expand_dims(alphas, axis=-1)  # [batch_size, max_len, node_size, 1]
    b = tf.expand_dims(betas, axis=2)    # [batch_size, max_len, 1, node_size]
    return a[:, 0:-1] + tf.reshape(trans_logps, shape=[1, 1, node_size, node_size]) + b[:, 1:]


# def get_gold_probs(labels, lengths):
#     batch_size = tf.shape(labels)[0]
#


def get_logsum(alphas, lengths):
    batch_size = tf.shape(alphas)[0]
    alpha_final = tf.gather_nd(alphas,
                               tf.stack([tf.range(0, batch_size), lengths-1], axis=1))  # [batch_size, node_size]
    return tf.reduce_logsumexp(alpha_final, axis=-1)



