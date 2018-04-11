import tensorflow as tf
import numpy as np


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
        n-1)


def tf_metropolis_sampling(n, probs, state, multiple_trial=1):
    sample_shape = tf.shape(probs)[0: -1]
    p = tf.reshape(probs, [-1, n])
    sample_num = tf.shape(p)[0]
    sample_range = tf.range(sample_num)

    y_multiple = tf.random_uniform(shape=[sample_num, multiple_trial])
    y_multiple = tf.minimum(tf.cast(y_multiple * n, dtype=tf.int32), n-1)
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

            outputs = tf.reshape(inputs, [-1, input_dim])                       # [batch_size * step_size, input_dim]
            logits = tf.matmul(outputs, softmax_w) + softmax_b                  # [batch_size * step_size, vocab_size]
            logits = tf.reshape(logits, tf.concat([input_shape, [vocab_size]], axis=0))   # (batch_size, step_size, vocab_size)

            if labels is not None:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                self._loss = tf.reduce_sum(loss) / tf.cast(input_shape[0], dtype=tf.float32)  # average over batch_size
                self._logps = - loss
            else:
                self._loss = tf.no_op()
                self._logps = tf.no_op()

            probs = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))  # (sample_num, vocab_size)
            draw = tf_random_fast_choice(probs, stride)     # (sample_num)
            draw = tf.reshape(draw, input_shape)            # (batch_size, step_size)
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
            inputs = tf.reshape(inputs, [-1, input_dim], name='list_inputs')  # to shape [batch_size * step_size, input_dim]
            logits = tf.matmul(inputs, softmax_w) + softmax_b

            if labels is not None:
                labels = tf.reshape(labels, [-1], name='list_labels')  # [batch_size * step_size]

                mask = tf.greater_equal(labels, shortlist[0])
                short_labels = tf.where(mask, shortlist[0] * tf.ones_like(labels), labels)  # set all the word >= shortlist to shortlist
                head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=short_labels, name='head_loss')

                tail_loss = tf.where(mask,
                                     np.log(shortlist[1]-shortlist[0]) * tf.ones_like(head_loss),
                                     tf.zeros_like(head_loss))

                loss = tf.reshape(head_loss + tail_loss, [batch_size, -1])
                self._loss = tf.reduce_sum(self._loss) / tf.cast(tf.shape(inputs)[0], dtype=tf.float32)  # average over batch_size
                self._logps = -loss
            else:
                self._loss = tf.no_op('no_loss_for_None_labels')
                self._logps = tf.no_op('no_logps_for_None_labels')

            # create sampling operation
            probs = tf.nn.softmax(logits)  # of shape (sample_num, vocab_size)
            head_draw = tf_random_fast_choice(probs, stride)
            head_logp = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=head_draw)

            mask = tf.greater_equal(head_draw, vocab_size-1)
            tail_draw = tf.where(mask,
                                 tf_random_int(tf.shape(head_draw), shortlist[1]-shortlist[0]),
                                 tf.zeros_like(head_draw))
            tail_logp = tf.where(mask,
                                 -np.log(shortlist[1]-shortlist[0]) * tf.ones_like(head_logp),
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

            cluster_num = len(cutoff) - 1           # the tail cluster number
            head_dim = cutoff[0] + cluster_num      # the head word + tail cluster number

            tail_project_factor = project_factor
            tail_w = []
            for i in range(cluster_num):
                project_dim = max(1, input_dim // tail_project_factor)
                tail_dim = cutoff[i+1] - cutoff[i]
                tail_w.append([
                    tf.get_variable('tail_{}_proj_w'.format(i+1),
                                    [input_dim, project_dim], initializer=initializer),
                    tf.get_variable('tail_{}_w'.format(i+1),
                                    [project_dim, tail_dim], initializer=initializer)
                ])
                tail_project_factor *= project_factor

            # get tail masks and update head labels
            loss_list = []
            head_labels = labels
            for i in range(cluster_num):
                # set ture if current sample belonging to class i
                mask = tf.logical_and(tf.greater_equal(labels, cutoff[i]),
                                      tf.less(labels, cutoff[i+1]))

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

            self._loss = tf.reshape(tf.add_n(loss_list), [batch_size, -1])
            self._loss = tf.reduce_sum(self._loss) / tf.cast(batch_size, dtype=tf.float32)

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
        w = tf.get_variable(name+'_w', [input_dim, output_dim], dtype=tf.float32, trainable=trainable)
        outputs = tf.matmul(tf.reshape(inputs, [-1, input_dim]), w)
        if use_bias:
            b = tf.get_variable(name+'_b', [output_dim], dtype=tf.float32, trainable=trainable,
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
        return i+1, final_seqs

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

        a_expand = tf.expand_dims(a, axis+1)
        multiples = [1] * (ndim+1)
        multiples[axis+1] = n
        a_tile = tf.tile(a_expand, multiples)

        res_shape = tf.concat([tf.shape(a)[0: axis], [-1], tf.shape(a)[axis+1:]], axis=0)
        a_repeat = tf.reshape(a_tile, res_shape)

        return a_repeat


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
            else:
                print('[layers.TrainOp] input loss, opt_method=%s' % optimize_method)
                grads = tf.gradients(loss, tvars)
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
        session.run(self._train_op, feed_dict)


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












