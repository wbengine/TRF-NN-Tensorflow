import tensorflow as tf
import numpy as np

from base import reader


def tf_random_choice(n, p, dtype=tf.int32):
    """
    using tensorflow to reslize the 1-d sample, similar to numpy.random.choich(a, p=p)
    Args:
        n: the sample size
        p: a 'Tensor' of shape [batch_size, sampling_size],
            i.e. several 1-d problities need to sample from
        dtype: tf.int32 (default) or tf.int64

    Returns:
        a 'Tensor' of shape [batch_size]
    """
    a = tf.cumsum(p, axis=-1)
    k = tf.expand_dims(tf.random_uniform(shape=tf.shape(a)[0:-1]), -1)
    draw = tf.cast(tf.reduce_sum(tf.sign(k - a) + 1, axis=-1), dtype=dtype) // 2
    draw = tf.minimum(draw, n - 1)
    return draw


def tf_random_fast_choice(p, stride, dtype=tf.int32):
    """
    a fast version of tf_random_choice, with a parameter stride
    Args:
        p: a 'Tensor' of shape [batch_size, sampling_size],
            i.e. several 1-d problities need to sample from.
            The last Dim shouled be known
        stride: int, <= sampling_size,
            denoting the sample stride
        dtype: tf.int32 (default) or tf.int54

    Returns:
        a 'Tensor' of shape [batch_size]
    """
    n = p.shape[-1].value  # sample space
    cluster_num = int(np.ceil(1.0 * n / stride))
    pad_num = cluster_num * stride - n

    pad_p = tf.pad(p, [[0, 0], [0, pad_num]], mode="CONSTANT")
    pad_p = tf.expand_dims(pad_p, axis=-1)
    pad_p = tf.reshape(pad_p, [-1, cluster_num, stride])
    p_cluster = tf.reduce_sum(pad_p, axis=-1)
    p_token = pad_p / tf.expand_dims(p_cluster, axis=-1)

    # draw sample
    draw_c = tf_random_choice(cluster_num, p=p_cluster, dtype=dtype)
    mask = tf.equal(tf.one_hot(draw_c, cluster_num, dtype=tf.int32), 1)
    p_given_c = tf.boolean_mask(p_token, mask)  # of shape (batch_size, stride)
    draw = tf_random_choice(stride, p=p_given_c, dtype=dtype)
    draw = draw_c * stride + draw

    return draw


def tf_random_int(shape, n):
    """
    generate random int in [0, n)
    """
    return tf.minimum(
        tf.cast(tf.random_uniform(shape=shape) * n, tf.int32),
        n - 1)


def tf_metropolis_sampling(n, probs, state, multiple_trial=1):
    sample_shape = tf.shape(probs)[0: -1]
    p = tf.reshape(probs, [-1, n])
    sample_num = tf.shape(p)[0]
    sample_range = tf.range(sample_num)

    y_multiple = tf.random_uniform(shape=[sample_num, multiple_trial])
    y_multiple = tf.minimum(tf.cast(y_multiple * n, dtype=tf.int32), n - 1)
    # [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
    range_idx = tf.reshape(tf.tile(tf.expand_dims(sample_range, axis=-1), [1, multiple_trial]), [-1])
    # [[0, y00], [0, y01], [0, y02], [1, y10], [1, y11], [1, y12],...]
    sample_idx = tf.stack([range_idx, tf.reshape(y_multiple, [-1])], axis=1)
    # gather the probability
    p_y_multiple = tf.gather_nd(p, sample_idx)
    p_y_multiple = tf.reshape(p_y_multiple, [-1, multiple_trial])
    w_y_multiple = p_y_multiple
    # sum the p(y_j)
    sum_w = tf.reduce_sum(w_y_multiple, axis=-1)

    # draw y form w_y_multiple
    w_y_multiple_norm = w_y_multiple / tf.reduce_sum(w_y_multiple, axis=-1, keep_dims=True)
    y_idx = tf_random_choice(multiple_trial, w_y_multiple_norm)
    y_idx = tf.stack([sample_range, y_idx], axis=1)
    y = tf.gather_nd(y_multiple, y_idx)
    w_y = tf.gather_nd(w_y_multiple, y_idx)

    # compute w_x
    x_idx = tf.stack([sample_range, tf.reshape(state, [-1])], axis=1)
    p_x = tf.gather_nd(p, x_idx)
    w_x = p_x

    # acceptance rate
    acc_prob = tf.minimum(1., sum_w / (sum_w - w_y + w_x))
    accept = tf.greater_equal(acc_prob, tf.random_uniform([sample_num]))
    res = tf.where(accept, y, state)
    res = tf.reshape(res, sample_shape)

    return res, accept


class Softmax(object):
    def __init__(self, inputs, labels, vocab_size, stride=None, name='Softmax',
                 weight_w=None, weight_b=None):
        """

        Args:
            inputs: 'Tensor' of shape [batch_size, step_size, dim]
            labels: 'Tensor' of shape [batch_size, step_size]. If None, donot compute the loss
            vocab_size: vocabulary size
            stride: a int32, used to accelearte the sampling, default=np.sqrt(vocab_size)
            name: name of name_scope
            weight_w: the weight. If None, create the variables using get_variables
            weight_b: the bias. If None, create the bias using get_variables; If 0, remove the bias
        """
        self.vocab_size = vocab_size
        input_shape = tf.shape(inputs)[0:-1]
        input_dim = inputs.shape[-1].value

        if stride is None:
            stride = int(np.sqrt(vocab_size))

        with tf.name_scope(name):
            if weight_w is None:
                softmax_w = tf.get_variable(name + '/w', [input_dim, vocab_size], dtype=tf.float32)
            else:
                if weight_w.shape != [input_dim, vocab_size]:
                    raise TypeError('[Softmax] the input weight_w shape {} is not equal to the need shape {}'.format(
                        weight_w.shape, [input_dim, vocab_size]))
                softmax_w = weight_w

            if weight_b is None:
                softmax_b = tf.get_variable(name + '/b', [vocab_size], dtype=tf.float32)
            else:
                softmax_b = weight_b  # the weight_b can be constant, such as 0

            outputs = tf.reshape(inputs, [-1, input_dim])  # [batch_size * step_size, input_dim]
            logits = tf.matmul(outputs, softmax_w) + softmax_b  # [batch_size * step_size, vocab_size]
            logits = tf.reshape(logits,
                                tf.concat([input_shape, [vocab_size]], axis=0))  # (batch_size, step_size, vocab_size)

            if labels is not None:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                self._loss = tf.reduce_sum(loss) / tf.cast(input_shape[0], dtype=tf.float32)  # average over batch_size
                self._logps = - loss
            else:
                self._loss = tf.no_op()
                self._logps = tf.no_op()

            probs = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))  # (sample_num, vocab_size)
            draw = tf_random_fast_choice(probs, stride)  # (sample_num)
            draw = tf.reshape(draw, input_shape)  # (batch_size, step_size)
            draw_nlogp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=draw)
            self._draw = [draw, -draw_nlogp]

            # self._probs = probs[:, 0]
            self._probs = tf.reshape(probs, tf.concat([input_shape, [vocab_size]], axis=0))

            # self.my_loss = tf.gather_nd(probs, tf.transpose(tf.stack([tf.range(tf.shape(probs)[0]), tf.reshape(labels, [-1])])))

    @property
    def loss(self):
        """
        loss, used to compute the gradient
        """
        return self._loss

    @property
    def draw(self):
        """
        return a list of [draw_idx, draw_nlogp]
        Returns:

        """
        return self._draw

    @property
    def logps(self):
        """
        return the log probs, at each position of each batch
        """
        return self._logps

    @property
    def probs(self):
        """
        return the probs, at each position of each batch
        """
        return self._probs


class ShortlistSoftmax(object):
    def __init__(self, inputs, labels, shortlist, stride=None, name=None):
        """
        Create a softmax with shortlist

        Args:
            inputs: 'Tensor' of shape [batch_size, step_size, dim]
            labels: 'Tensor' of shape [batch_size, step_size]
            shortlist: a list of two values,
                the first value indicates the shortlist,
                and the last value indicate the vocab_size,
                such as [2000, vocab_size]
            stride: a int32, used to accelearte the sampling
            name: the name_scope, (can be Nont)
        """
        vocab_size = shortlist[0] + 1
        batch_size = tf.shape(inputs)[0]
        input_dim = inputs.shape[-1].value
        if stride is None:
            stride = int(np.sqrt(vocab_size))

        with tf.name_scope(name, 'ShortlistSoftmax'):
            softmax_w = tf.get_variable('%s/w' % name, [input_dim, vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable('%s/b' % name, [vocab_size], dtype=tf.float32)
            inputs = tf.reshape(inputs, [-1, input_dim],
                                name='list_inputs')  # to shape [batch_size * step_size, input_dim]
            logits = tf.matmul(inputs, softmax_w) + softmax_b

            if labels is not None:
                labels = tf.reshape(labels, [-1], name='list_labels')  # [batch_size * step_size]

                mask = tf.greater_equal(labels, shortlist[0])
                short_labels = tf.where(mask, shortlist[0] * tf.ones_like(labels),
                                        labels)  # set all the word >= shortlist to shortlist
                head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=short_labels,
                                                                           name='head_loss')

                tail_loss = tf.where(mask,
                                     np.log(shortlist[1] - shortlist[0]) * tf.ones_like(head_loss),
                                     tf.zeros_like(head_loss))

                loss = tf.reshape(head_loss + tail_loss, [batch_size, -1])
                self._loss = tf.reduce_sum(loss) / tf.cast(tf.shape(inputs)[0],
                                                           dtype=tf.float32)  # average over batch_size
                self._logps = -loss
            else:
                self._loss = tf.no_op('no_loss_for_None_labels')
                self._logps = tf.no_op('no_logps_for_None_labels')

            # create sampling operation
            probs = tf.nn.softmax(logits)  # of shape (sample_num, vocab_size)
            head_draw = tf_random_fast_choice(probs, stride)
            head_logp = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=head_draw)

            mask = tf.greater_equal(head_draw, vocab_size - 1)
            tail_draw = tf.where(mask,
                                 tf_random_int(tf.shape(head_draw), shortlist[1] - shortlist[0]),
                                 tf.zeros_like(head_draw))
            tail_logp = tf.where(mask,
                                 -np.log(shortlist[1] - shortlist[0]) * tf.ones_like(head_logp),
                                 tf.zeros_like(head_logp))

            draw = head_draw + tail_draw
            draw_logp = head_logp + tail_logp
            self._draw = [tf.reshape(draw, [batch_size, -1]),
                          tf.reshape(draw_logp, [batch_size, -1])
                          ]

    @property
    def loss(self):
        """
        return the loss of softmax
        Returns:
            a 'Tensor' of shape (batch_size, step_size)
        """
        return self._loss

    @property
    def draw(self):
        return self._draw

    @property
    def logps(self):
        return self._logps


class AdaptiveSoftmax(object):
    def __init__(self, inputs, labels, cutoff, project_factor=4., initializer=None, name=None):
        """
        Create a AdaptiveSoftmax class

        Args:
            inputs: A 'Tensor' of shape [batch_size, step_size, dim]
            labels: A 'Tensor' of shape [batch_size, step_size]
            cutoff: A list indicating the limits of each clusters, such as [2000, vocab_size]
            project_factor:
            initializer:
            name:
        """
        with tf.name_scope(name, 'AdaptiveSoftmax'):

            self._dbg = []

            batch_size = inputs.shape[0].value
            input_dim = inputs.shape[-1].value

            # flatten the data and labels
            input_dim = inputs.shape[-1].value  # the input_dim should be konwn
            inputs = tf.reshape(inputs, [-1, input_dim], name='inputs')
            labels = tf.reshape(labels, [-1], name='labels')

            cluster_num = len(cutoff) - 1  # the tail cluster number
            head_dim = cutoff[0] + cluster_num  # the head word + tail cluster number

            tail_project_factor = project_factor
            tail_w = []
            for i in range(cluster_num):
                project_dim = max(1, input_dim // tail_project_factor)
                tail_dim = cutoff[i + 1] - cutoff[i]
                tail_w.append([
                    tf.get_variable('tail_{}_proj_w'.format(i + 1),
                                    [input_dim, project_dim], initializer=initializer),
                    tf.get_variable('tail_{}_w'.format(i + 1),
                                    [project_dim, tail_dim], initializer=initializer)
                ])
                tail_project_factor *= project_factor

            # get tail masks and update head labels
            loss_list = []
            head_labels = labels
            for i in range(cluster_num):
                # set ture if current sample belonging to class i
                mask = tf.logical_and(tf.greater_equal(labels, cutoff[i]),
                                      tf.less(labels, cutoff[i + 1]))

                # update head labels
                head_labels = tf.where(mask,
                                       (cutoff[0] + i) * tf.ones_like(head_labels),
                                       head_labels)

                # compute tail loss
                tail_inputs = inputs
                tail_labels = tf.where(mask, labels - cutoff[i], tf.zeros_like(labels))
                tail_logits = tf.matmul(tf.matmul(tail_inputs, tail_w[i][0]), tail_w[i][1])
                tail_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tail_logits, labels=tail_labels)

                # as tail_loss is of shape [k_i] and k_i <= sample_num
                # algined_tail_loss = tf.sparse_to_dense(tf.where(mask),
                #                                        tf.cast(tf.shape(labels), tf.int64),
                #                                        tail_loss)
                algined_tail_loss = tail_loss * tf.cast(mask, tf.float32)
                loss_list.append(algined_tail_loss)

                self._dbg.append(mask)
                self._dbg.append(tail_inputs)
                self._dbg.append(tail_labels)
                self._dbg.append(tail_logits)
                self._dbg.append(tail_loss)
                self._dbg.append(algined_tail_loss)

            # compute the head loss
            head_w = tf.get_variable('head_w', [input_dim, head_dim], initializer=initializer)
            head_logits = tf.matmul(inputs, head_w)
            head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=head_logits, labels=head_labels)
            loss_list.append(head_loss)

            loss = tf.reshape(tf.add_n(loss_list), [batch_size, -1])
            self._loss = tf.reduce_sum(loss) / tf.cast(batch_size, dtype=tf.float32)

            self._logps = - loss
            # define the sample operation

    @property
    def loss(self):
        """
        return the loss of softmax
        Returns:
            a 'Tensor' of shape (batch_size, step_size)
        """
        return self._loss

    @property
    def logps(self):
        return self._logps

    @property
    def pred(self):
        """
        return the pred of softmax, used to sample
        Returns:
            a 'Tensor'
        """
        return self._pred


class SummaryScalarBank(object):
    def __init__(self, name_list, name_scope='scalar_bank'):
        """
        init a scalar bank

        Args:
            name_list: a list of string indicating the name of each scalar, such as
                ['ppl', 'nll', 'step', 'learnrate']. The default type is tf.float32
            name_scope: a collection name (string)
        """
        self.name_scope = name_scope

        with tf.name_scope(name_scope):
            self._placeholder_float32 = tf.placeholder(dtype=tf.float32, shape=[])
            self.scalar_dict = {}
            for w in name_list:
                self.scalar_dict[w] = tf.summary.scalar(w, self._placeholder_float32,
                                                        collections=['summary_scalar'])

    def add(self, name):
        if name not in self.scalar_dict:
            with tf.name_scope(self.name_scope):
                self.scalar_dict[name] = tf.summary.scalar(name, self._placeholder_float32,
                                                           collections=['summary_scalar'])

    def write_summary(self, sv, sess, name, value, global_step=None):
        summ = sess.run(self.scalar_dict[name], {self._placeholder_float32: value})
        sv.summary_computed(sess, summ, global_step)


class SummaryVariables(object):
    def __init__(self, tvars=None):
        if tvars is None:
            tvars = tf.trainable_variables()
        for t in tvars:
            tf.summary.histogram(name=t.name, values=t, collections=['summary_variables'])
        self.summ = tf.summary.merge_all(key='summary_variables')

    def write_summary(self, sv, sess, global_step=None):
        sv.summary_computed(sess, sess.run(self.summ), global_step=global_step)


def batch_pick(params, indexs, batch_size=None):
    """
    pick each elements of each batch data

    Args:
        params: tensor of shape [batch_size, seq_len, B, ...]
        indexs: 1-D tensor of int32
        batch_size: the batch_size of params

    Returns:
        for i in range(batch_size):
            res[i] = params[i, indexs[i]]
        return res
    """
    if batch_size is None:
        batch_size = tf.shape(params)[0]

    ii = tf.stack([tf.range(0, batch_size, dtype=tf.int32), indexs], axis=1)
    return tf.gather_nd(params, ii)


def embedding_mask(embs, lengths):
    """
    mask the embedding based the length of each sequence
    Args:
        embs: 3-D tensor, indicating the embedding of each sequence
        lengths: 1-D tensor, indicating the length of each sequence

    Returns:
        a 3-D tensor, all the embedding over the length is set to zeros
    """
    batch_size = tf.shape(embs)[0]
    max_len = tf.shape(embs)[1]
    len_mask = tf.tile(tf.reshape(tf.range(max_len, dtype=tf.int32), [1, max_len]),
                       [batch_size, 1])
    len_mask = tf.less(len_mask, tf.reshape(lengths, [batch_size, 1]))
    len_mask = tf.cast(len_mask, tf.float32)  # shape (batch_size, max_len)
    len_mask = tf.expand_dims(len_mask, axis=-1)  # shape (batch_size, max_len, 1)
    return embs * len_mask


def linear(inputs, output_dim, activate=None, use_bias=True, name='linear', trainable=True):
    """
    perform a linear layer y = Wx + b
    Args:
        inputs: 3-D Tensor, [batch_size, length, input_dim], the input x
        output_dim: int, the output dim
        use_bias: bool, using the bias
        name: the name of this operation
        activate: the activation function
        trainable: True or False

    Returns:
        the output y
    """
    with tf.name_scope(name):
        input_dim = inputs.shape[-1].value
        w = tf.get_variable(name + '_w', [input_dim, output_dim], dtype=tf.float32, trainable=trainable)
        outputs = tf.matmul(tf.reshape(inputs, [-1, input_dim]), w)
        if use_bias:
            b = tf.get_variable(name + '_b', [output_dim], dtype=tf.float32, trainable=trainable,
                                initializer=tf.zeros_initializer())
            outputs += b
        if activate is not None:
            outputs = activate(outputs)
        outputs = tf.reshape(outputs, tf.concat([tf.shape(inputs)[0:-1], [output_dim]], axis=0))
    return outputs


def generate_length_mask(lengths, batch_size, max_len, dtype=tf.float32):
    """
    generate the length mask
    Args:
        lengths: 1-d Tensor indicating the length [batch_size]
        batch_size: the batch size
        max_len: the maximum length
        dtype: the type of values

    Returns:
        (2-D length mask, 3-D length mask)
    """
    len_mask = tf.tile(tf.reshape(tf.range(max_len, dtype=tf.int32), [1, max_len]), [batch_size, 1])
    len_mask = tf.less(len_mask, tf.reshape(lengths, [batch_size, 1]))
    len_mask = tf.cast(len_mask, dtype)  # shape (batch_size, max_len)
    expand_len_mask = tf.expand_dims(len_mask, axis=-1)  # shape (batch_size, max_len, 1)

    return len_mask, expand_len_mask


def concate_sequences(seqs1, lengths1, seqs2, lengths2):
    """
    concatenate two sequences with different lengths
    such as:
        seqs1 = [[1,1,0],[2,2,2]], lengths1 = [2, 3]
        seqs2 = [[3,0,0],[4,4,0]], lengths2 = [1, 2]
    return:
        seqs = [[1,1,3,0,0], [2,2,2,4,4]], whose lengths is [3, 5] with 0 padded
    """
    final_lens = lengths1 + lengths2
    maxfinallen = tf.reduce_max(final_lens)

    batch_size = tf.shape(seqs1)[0]

    def body(i, final_seqs):
        s1 = seqs1[i, 0: lengths1[i]]
        s2 = seqs2[i, 0: lengths2[i]]
        ss = tf.concat([s1, s2, tf.zeros([maxfinallen - lengths1[i] - lengths2[i]], dtype=seqs1.dtype)], axis=0)
        final_seqs = tf.concat([final_seqs, tf.reshape(ss, [1, -1])], axis=0)
        return i + 1, final_seqs

    def cond(i, _):
        return i < batch_size

    _, seqs = tf.while_loop(cond, body,
                            loop_vars=[0,
                                       tf.zeros([0, maxfinallen], dtype=seqs1.dtype)],
                            shape_invariants=[tf.TensorShape([]),
                                              tf.TensorShape([None, None])])
    return seqs


def append_sequence(seqs, lengths, append_values, append_len=1, name='append_sequence'):
    """
    For example:
        seqs = [[0, 0, x], [1, 1, 1]],
        lengths = [2, 3]
        append_values = 4
        append_len = 2
    return:
        [[0, 0, 4, 4, x],
         [1, 1, 1, 4, 4]]
        x denotes unknown values
    """
    with tf.name_scope(name):
        if append_len == 1:
            batch_size = tf.shape(seqs)[0]
            seqs_pad = tf.pad(seqs, [[0, 0], [0, 1]])
            add_index = tf.stack([tf.range(batch_size), lengths], axis=1)
            seqs_add = tf.scatter_nd(add_index,
                                     append_values * tf.ones([batch_size], dtype=seqs.dtype),
                                     tf.shape(seqs_pad))
            seqs_msk = tf.scatter_nd(add_index,
                                     tf.ones([batch_size], dtype=seqs.dtype),
                                     tf.shape(seqs_pad))
            return seqs_pad * (1 - seqs_msk) + seqs_add
        else:
            return concate_sequences(seqs, lengths,
                                     append_values * tf.ones([tf.shape(seqs)[0], append_len], dtype=seqs.dtype),
                                     append_len * tf.ones_like(lengths))


def repeat(a, n, axis, name='repeat'):
    """repeat a tensor along the axis.
    For expamle:
        a=[0, 1, 2, 3], n=2, axis=0
        return [0, 0, 1, 1, 2, 2, 3, 3]
    """
    with tf.name_scope(name):
        ndim = len(a.shape)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise TypeError('[repeat] axis={} not match the ndim={} of tansor'.format(axis, ndim))

        a_expand = tf.expand_dims(a, axis + 1)
        multiples = [1] * (ndim + 1)
        multiples[axis + 1] = n
        a_tile = tf.tile(a_expand, multiples)

        res_shape = tf.concat([tf.shape(a)[0: axis], [-1], tf.shape(a)[axis + 1:]], axis=0)
        a_repeat = tf.reshape(a_tile, res_shape)

        return a_repeat


def rnn(inputs, lengths, rnn_hidden_size, rnn_hidden_layers, rnn_type,
        dropout=None, reuse=None, name='rnn',
        init_states=None):
    """
    create rnn module
    Args:
        inputs: tensor, [batch_size, max_lengths]
        lengths: tensor, [batch_size]
        rnn_hidden_size: int
        rnn_hidden_layers: int
        rnn_type: one of 'rnn', 'lstm', 'brnn', 'blstm'
        dropout: None, or float
        reuse: None, or True
        name: str.
        init_states: None or tensor

    Returns:
        outputs, states

        if one-directional rnn, outputs = tensor of [batch_size, max_length, hidden_size]
        if bi-directional rnn,  outputs = tuple( forward_outputs, backward_outputs )
    """

    # rnn cell
    def one_lstm_cell():
        if rnn_type.lower().find('lstm') != -1:
            c = tf.contrib.rnn.BasicLSTMCell(rnn_hidden_size, forget_bias=0., reuse=reuse)
        elif rnn_type.lower().find('rnn') != -1:
            c = tf.contrib.rnn.BasicRNNCell(rnn_hidden_size, activation=tf.nn.tanh, reuse=reuse)
        else:
            raise TypeError('undefined rnn type = ' + rnn_type)
        if dropout is not None and dropout > 0:
            c = tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=1. - dropout)
        return c

    with tf.name_scope(name):
        # recurrent structure
        if rnn_type[0].lower() == 'b':
            cell_fw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(rnn_hidden_layers)])
            cell_bw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(rnn_hidden_layers)])
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                              inputs=inputs,
                                                              sequence_length=lengths,
                                                              dtype=tf.float32,
                                                              scope=name + '_brnn')
            outputs_fw = outputs[0]
            outputs_bw = outputs[1]
            final_outputs = (outputs_fw, outputs_bw)

        else:
            cell_fw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(rnn_hidden_layers)])
            if init_states is None:
                init_states = cell_fw.zero_state(tf.shape(inputs)[0], tf.float32)
            outputs, states = tf.nn.dynamic_rnn(cell_fw,
                                                inputs=inputs,
                                                sequence_length=lengths,
                                                initial_state=init_states)
            final_outputs = outputs

    return final_outputs, states


def char_emb_cnn(inputs, char_size, embedding_size,
                 cnn_kernel_size, cnn_kernel_width,
                 word_to_chars, reuse=None):
    """
    Args:
        inputs: tensor of int32, [batch_size, max_len]
        char_size: int, char_vocabulary_size
        embedding_size: int
        cnn_kernel_size: int / a list of int
        cnn_kernel_width: int / a list of int
        word_to_chars: a list of list
        reuse: reuse

    Returns:
        tensor, of shape [batch_size, max_len, output_dim]
          output_dim = sum(cnn_kernal_size)
    """
    with tf.name_scope('char_emb_cnn'):
        batch_size = tf.shape(inputs)[0]
        max_length = tf.shape(inputs)[1]

        char_arys, char_lens = reader.produce_data_to_array(word_to_chars)
        char_max_len = char_arys.shape[1]
        char_arys = tf.constant(char_arys, name='char_arys')
        char_lens = tf.constant(char_lens, name='char_lens')

        char_inputs = tf.gather(char_arys, tf.reshape(inputs, [-1]))  # [word_num, char_max_num]
        char_lens = tf.gather(char_lens, tf.reshape(inputs, [-1]))  # [word_num]
        char_mask = tf.sequence_mask(char_lens, maxlen=char_max_len, dtype=tf.float32)  # [word_num, char_max_len]
        char_mask_ext = tf.expand_dims(char_mask, axis=-1)  # [word_num, char_max_len, 1]

        # embedding
        char_embedding = tf.get_variable('char_embedding', [char_size, embedding_size], dtype=tf.float32)
        emb = tf.nn.embedding_lookup(char_embedding, char_inputs)  # (word_num, char_max_len, char_emb_dim)
        emb *= char_mask_ext

        # CNN
        if isinstance(cnn_kernel_width, int):
            cnn_kernel_width = [cnn_kernel_size]
        if isinstance(cnn_kernel_size, int):
            cnn_kernel_size = [cnn_kernel_size] * len(cnn_kernel_width)


        conv_list = []
        for w, s in zip(cnn_kernel_width, cnn_kernel_size):
            conv = tf.layers.conv1d(
                inputs=emb,  # set the values at positon >= length to zeros
                filters=s,
                kernel_size=w,
                padding='same',
                activation=tf.nn.relu,
                reuse=reuse,
                name='cnn%d' % w
            )
            conv_list.append(conv)

        conv = tf.concat(conv_list, axis=-1) * char_mask_ext  # (word_num, char_max_len, dim)

        # max-pooling
        outputs = tf.reduce_max(conv, axis=1)  # (word_num, dim)
        outputs = tf.reshape(outputs, [batch_size, max_length, sum(cnn_kernel_size)])  # (batch_size, max_len, dim)

    return outputs


def char_emb_rnn(inputs, char_size, embedding_size,
                 rnn_hidden_size, rnn_hidden_layers,
                 word_to_chars, reuse=None):
    """
    Using the BLSTM to achieve the char-level embedding of each words
    Args:
        inputs: Tensor of int32, [batch_size, max_len]
        char_size: int
        embedding_size: int
        rnn_hidden_size: int
        rnn_hidden_layers: int
        word_to_chars: a list of list
        reuse: None or True

    Returns:
        Tensor, of shape [batch_size, max_len, output_dim]
          output_dim = sum(cnn_kernal_size)
    """
    with tf.name_scope('char_emb_rnn'):
        batch_size = tf.shape(inputs)[0]
        max_length = tf.shape(inputs)[1]

        char_arys, char_lens = reader.produce_data_to_array(word_to_chars)
        char_max_len = char_arys.shape[1]
        char_arys = tf.constant(char_arys, name='char_arys')
        char_lens = tf.constant(char_lens, name='char_lens')

        char_inputs = tf.gather(char_arys, tf.reshape(inputs, [-1]))  # [word_num, char_max_num]
        char_lens = tf.gather(char_lens, tf.reshape(inputs, [-1]))  # [word_num]
        char_mask = tf.sequence_mask(char_lens, maxlen=char_max_len, dtype=tf.float32)  # [word_num, char_max_len]
        char_mask_ext = tf.expand_dims(char_mask, axis=-1)  # [word_num, char_max_len, 1]

        # embedding
        char_embedding = tf.get_variable('char_embedding', [char_size, embedding_size], dtype=tf.float32)
        emb = tf.nn.embedding_lookup(char_embedding, char_inputs)  # (word_num, char_max_len, char_emb_dim)
        emb *= char_mask_ext

        # blstm
        rnn_outputs, _ = rnn(emb, char_lens,
                             rnn_hidden_size=rnn_hidden_size,
                             rnn_hidden_layers=rnn_hidden_layers,
                             rnn_type='blstm',
                             reuse=reuse,
                             name='c2w_rnn')  # ([word_num, char_max_len, rnn_outputs], )
        rnn_outputs_fw = rnn_outputs[0]
        rnn_outputs_bw = rnn_outputs[1]

        emb_fw = tf.gather_nd(rnn_outputs_fw,
                              tf.stack([tf.range(tf.shape(rnn_outputs_fw)[0]), char_lens-1], axis=1))
        emb_bw = rnn_outputs_bw[:, 0]
        outputs = tf.concat([emb_fw, emb_bw], axis=-1)
        outputs = tf.reshape(outputs, [batch_size, max_length, rnn_hidden_size * 2])  # (batch_size, max_len, dim)

    return outputs


class TrainOp(object):
    def __init__(self, loss, tvars,
                 optimize_method='sgd', max_grad_norm=None,
                 initial_lr=1.0,
                 l2_reg=0,
                 name='train_op'):
        """
        Args:
            loss:  a Tensor, denoting the loss, or a list, denoting the gradient
            tvars: a Tensor or a list of tensor, denoting the variables
            optimize_method: string
            max_grad_norm: float32 or None
            initial_lr: learning rate
            name: name of this operations
        """
        self.tvars = tvars
        self.opt_method = optimize_method

        with tf.name_scope(name, 'train_op'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            # update lr
            self._lr = tf.Variable(initial_lr, trainable=False, name='lr')
            self._new_lr = tf.placeholder(dtype=tf.float32, shape=[], name='new_lr')
            self._update_lr = tf.assign(self._lr, self._new_lr)

            # compute the gradient
            if isinstance(loss, list):
                print('[layers.TrainOp] input gradient, opt_method=%s' % optimize_method)
                grads = loss
                self.loss = None
            else:
                print('[layers.TrainOp] input loss, opt_method=%s' % optimize_method)
                grads = tf.gradients(loss, tvars)
                self.loss = loss
            if max_grad_norm is not None:
                grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

            # train
            if optimize_method.lower() == 'adam':
                optimizer = tf.train.AdamOptimizer(self._lr)
            elif optimize_method.lower() == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self._lr)
            elif optimize_method.lower() == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self._lr, momentum=0.9)
            elif optimize_method.lower() == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self._lr)
            else:
                raise TypeError('[layers.TrainOp] undefined opt_method={}'.format(optimize_method))

            grads_reg = []
            for g, v in zip(grads, tvars):
                if l2_reg > 0:
                    g += l2_reg * v
                grads_reg.append(g)

            self._train_op = optimizer.apply_gradients(zip(grads_reg, tvars), global_step=self.global_step)

    @property
    def train_op(self):
        return self._train_op

    def set_lr(self, session, learning_rate):
        """set the learning rate"""
        session.run(self._update_lr, {self._new_lr: learning_rate})

    def update(self, session, feed_dict=None):
        if self.loss is None:
            session.run(self._train_op, feed_dict)
            return None
        else:
            v = session.run([self._train_op, self.loss], feed_dict)
            return v[1]  # return loss


def logaddexp(a, b):
    """
    compute log(exp(a) + exp(b))
    Args:
        a: a tensor
        b: a tensor

    Returns:
        log(exp(a) + exp(b))
    """
    return tf.reduce_logsumexp(tf.stack([a, b]), axis=0)


def logsubexp(a, b):
    c = tf.maximum(a, b)
    return c + tf.log(tf.exp(a - c) - tf.exp(b - c))


def log1mexp(x):
    return logsubexp(tf.zeros_like(x), x)
