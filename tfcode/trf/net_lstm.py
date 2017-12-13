import tensorflow as tf
import numpy as np

from . import reader
from . import layers
from . import wblib as wb


class NetLSTM(object):
    class Config(wb.PPrintObj):
        def __init__(self, data):
            self.vocab_size = data.get_vocab_size()
            self.batch_size = 100  # used in training and evaluation, to restrict the input batch_size of input sequences
            self.step_size = 100    # used in training, to restrict the input sequence lengths
            self.embedding_size = 200
            self.hidden_size = 200
            self.hidden_layers = 1
            self.weight_tying = True  # tie the weight of input embedding and output embedding
            self.softmax_type = 'Softmax'  # can be 'Softmax' or 'AdaptiveSoftmax' or 'Shortlist'
            self.adaptive_softmax_cutoff = [2000, self.vocab_size]  # for AdaptiveSoftmax, a list with the last value is vocab_size
            self.adaptive_softmax_project_factor = 4.  # for AdaptiveSoftmax a float >= 1.0
            self.softmax_pred_token_per_cluster = int(np.sqrt(self.vocab_size))  # used to accelerate the sampling
            self.init_weight = 0.1
            self.optimize_method = 'sgd'  # one of 'sgd', 'adam', 'adagrad'
            self.max_grad_norm = 5.0      # grad_norm, can be None
            self.learning_rate = 1.0      # [learning rate] initial values
            self.dropout = 0.0

    def __init__(self, config, reuse=None):
        self.config = config
        self.name = 'net_lstm'
        self.default_initializer = tf.random_uniform_initializer(-self.config.init_weight, self.config.init_weight)

        # Create LSTM cell
        def one_lstm_cell():
            c = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0., reuse=reuse)
            if self.config.dropout > 0:
                c = tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=1. - self.config.dropout)
            return c

        self.cell = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.hidden_layers)])

        with tf.name_scope(self.name):
            self._inputs = tf.placeholder(tf.int32, [None, None], name='net_inputs')
            self._targets = tf.placeholder(tf.int32, [None, None], name='net_targets')
            batch_size = tf.shape(self._inputs)[0]
            self._lengths = tf.ones([batch_size], dtype=tf.int32) * tf.shape(self._inputs)[1]
            self._initial_state = self.cell.zero_state(batch_size, tf.float32)  # initial state

            self.loss, self.final_state = self.get_loss(self._initial_state, self._inputs, self._lengths,
                                                        self._targets, reuse=reuse)

            # draw samples
            self._sample_nums = tf.placeholder(tf.int32, shape=[None], name='sample_nums')
            self.draw = self.get_draw(self._initial_state, self._inputs, self._lengths, self._sample_nums)

            # compute conditional logprobs
            self._beg_positions = tf.ones([batch_size], dtype=tf.int32)
            self._end_positions = self._lengths
            self.logp = self.get_logp(self._initial_state, self._inputs,
                                      self._beg_positions, self._end_positions)

            # training operations
            self.train_op, self.train_state = self.get_update(self._initial_state, self._inputs, self._lengths)

        # variables
        with tf.variable_scope(self.name, reuse=True):
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            if reuse is None:
                print('variables in %s' % self.name)
                for v in self.vars:
                    print('\t', v.name, v.shape, v.device)

    def output(self, _initial_state, _inputs, _lengths=None, _targets=None):

        if _initial_state is None:
            _initial_state = self.cell.zero_state(tf.shape(_inputs)[0], tf.float32)

        # word embedding
        with tf.device('/cpu:0'):
            word_embedding = tf.get_variable('word_embedding',
                                             [self.config.vocab_size, self.config.embedding_size],
                                             dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(word_embedding, _inputs)

        # dropout
        if self.config.dropout > 0:
            inputs = tf.nn.dropout(inputs, keep_prob=1. - self.config.dropout)

        # lstm
        # using tf.nn.dynamic_rnn to remove the constrains of step size.
        outputs, state = tf.nn.dynamic_rnn(self.cell,
                                           inputs=inputs,
                                           sequence_length=_lengths,
                                           initial_state=_initial_state)

        # softmax
        if self.config.softmax_type == 'AdaptiveSoftmax':
            print('apply adaptiveSoftmax')
            softmax = layers.AdaptiveSoftmax(outputs, _targets,
                                             cutoff=self.config.adaptive_softmax_cutoff,
                                             project_factor=self.config.adaptive_softmax_project_factor,
                                             name='Softmax_Adaptive')
        elif self.config.softmax_type == 'Softmax':
            if self.config.weight_tying:
                softmax_w = tf.transpose(word_embedding)
                softmax_b = 0
            else:
                softmax_w = None
                softmax_b = None
            softmax = layers.Softmax(outputs, _targets, self.config.vocab_size,
                                     self.config.softmax_pred_token_per_cluster,
                                     weight_w=softmax_w,
                                     weight_b=softmax_b,
                                     name='Softmax')
        elif self.config.softmax_type == 'Shortlist':
            softmax = layers.ShortlistSoftmax(outputs, _targets,
                                              shortlist=self.config.adaptive_softmax_cutoff,
                                              stride=None,
                                              name='Softmax_Shortlist')
        else:
            raise TypeError('Unknown softmax method =' + self.config.softmax_type)

        return softmax, state

    def get_loss(self, _initial_state, _inputs, _lengths, _targets, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse, initializer=self.default_initializer):
            softmax, state = self.output(_initial_state, _inputs, _lengths, _targets)
        return softmax.loss, state

    def get_draw(self, _initial_state, _inputs, _lengths, _sample_nums, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse, initializer=self.default_initializer):

            def fn_body(i, state, draw_words, draw_logps):
                """
                loop body
                Args:
                    i: the rest sample number
                    state: input state
                    draw_words: returned words [batch_size, sample_num-i]
                    draw_logps: returned logps [batch_size, sample_num-i]

                Returns:
                    (i-1, final_state, samples, draw_words, draw_logps)
                """
                _softmax, _final_state = self.output(state, draw_words[:, -1:])
                new_draw_words = tf.concat([draw_words, _softmax.draw[0]], axis=-1)
                new_draw_logps = tf.concat([draw_logps, _softmax.draw[1]], axis=-1)
                return i-1, _final_state, new_draw_words, new_draw_logps

            def fn_cond(i, state, draw_words, draw_logps):
                return i > 0

            # forward, get the final state and the first samples
            softmax, final_state = self.output(_initial_state, _inputs[:, 0: tf.reduce_max(_lengths)], _lengths)
            tail_index = tf.stack([tf.range(tf.shape(_lengths)[0]), _lengths-1], axis=1)
            initial_draw_words = tf.reshape(tf.gather_nd(softmax.draw[0], tail_index), [-1, 1])  # [batch_size, sample_num-i]
            initial_draw_logps = tf.reshape(tf.gather_nd(softmax.draw[1], tail_index), [-1, 1])  # [batch_size, sample_num-i]
            # shape of rnn state
            state_shape = tuple([tf.contrib.rnn.LSTMStateTuple(c.get_shape(), h.get_shape()) for c, h in final_state])
            # draw max(sample_nums)-1 times
            _, final_state, draw_words, draw_logps = tf.while_loop(
                fn_cond, fn_body,
                loop_vars=[tf.reduce_max(_sample_nums)-1,
                           final_state,
                           initial_draw_words,
                           initial_draw_logps],
                shape_invariants=[tf.TensorShape([]),
                                  state_shape,
                                  tf.TensorShape([None, None]),
                                  tf.TensorShape([None, None])]
            )
            # concatenate the samples words with the initial sequences
            final_seqs = layers.concate_sequences(_inputs, _lengths, draw_words, _sample_nums)
            final_lengths = _lengths + _sample_nums
            # summary the sample logps
            draw_logp = tf.reduce_sum(draw_logps * tf.sequence_mask(_sample_nums, dtype=tf.float32), axis=-1)

        return final_seqs, final_lengths, draw_logp, final_state

    def get_state_gather(self, state, indices):
        """return the gathered state"""
        res_state = []
        for (c, h) in state:
            res_c = tf.gather(c, indices)
            res_h = tf.gather(h, indices)
            res_state.append(tf.contrib.rnn.LSTMStateTuple(res_c, res_h))
        return tuple(res_state)

    def get_state_operation(self, state, fun):
        """perform function fun to each c and h in states"""
        res_state = []
        for (c, h) in state:
            res_c = fun(c)
            res_h = fun(h)
            res_state.append(tf.contrib.rnn.LSTMStateTuple(res_c, res_h))
        return tuple(res_state)

    def get_state_repeat(self, state, repeat_num):
        res_state = []
        for (c, h) in state:
            res_c = tf.reshape(layers.repeat(c, repeat_num, axis=0), [-1, self.config.hidden_size])
            res_h = tf.reshape(layers.repeat(h, repeat_num, axis=0), [-1, self.config.hidden_size])
            res_state.append(tf.contrib.rnn.LSTMStateTuple(res_c, res_h))
        return tuple(res_state)

    def get_logp(self, _initial_state, _inputs, _beg_positions, _end_positions, reuse=True):
        """compute the conditional logp from beg_pos to end_pos, [beg_pos, end_pos)"""
        with tf.variable_scope(self.name, reuse=reuse, initializer=self.default_initializer):
            max_end = tf.reduce_max(_end_positions)
            softmax, final_state = self.output(_initial_state, _inputs[:, 0: max_end-1],
                                               _lengths=None, _targets=_inputs[:, 1: max_end])

            masks = tf.logical_and(tf.logical_not(tf.sequence_mask(_beg_positions-1, max_end-1)),
                                   tf.sequence_mask(_end_positions-1, max_end-1))
            masks = tf.cast(masks, tf.float32)
            logps = -tf.reduce_sum(softmax.loss * masks, axis=-1)

        return logps, final_state

    def get_update(self, _initial_state, _inputs, _lengths, reuse=True):
        """update the parameters"""
        with tf.variable_scope(self.name, reuse=reuse, initializer=self.default_initializer):
            maxlen = tf.reduce_max(_lengths)
            softmax, state = self.output(_initial_state, _inputs[:, 0:maxlen-1], _lengths-1, _inputs[:, 1:maxlen])

            mask = tf.sequence_mask(_lengths-1, maxlen-1, dtype=tf.float32)
            loss = tf.reduce_sum(softmax.loss * mask) / tf.reduce_sum(mask)  # average for each word

            # parameters
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            # training operations
            train_op = layers.TrainOp(loss, tvars, self.config.optimize_method,
                                      max_grad_norm=self.config.max_grad_norm,
                                      initial_lr=self.config.learning_rate)

        return train_op, state

    def run_train(self, session, inputs, lengths):
        # reshape the inputs
        seq_list = []
        for x, n in zip(inputs, lengths):
            for i in range(1, n, self.config.step_size):
                seq_list.append(x[i-1: i+self.config.step_size])
        inputs, lengths = reader.produce_data_to_trf(seq_list)

        for i in range(0, len(inputs), self.config.batch_size):
            session.run(self.train_op.train_op, {self._inputs: inputs[i: i+self.config.batch_size],
                                                 self._lengths: lengths[i: i+self.config.batch_size]})

    def run_eval(self, session, seq_list):
        logps = np.zeros(len(seq_list))
        for i in range(0, len(seq_list), self.config.batch_size):
            inputs, lengths = reader.produce_data_to_trf(seq_list[i: i+self.config.batch_size])
            logps[i: i+self.config.batch_size], _ = session.run(self.logp, {self._inputs: inputs, self._lengths: lengths})

        lengths = [len(s) for s in seq_list]
        words = np.sum(lengths) - len(seq_list)
        sents = len(seq_list)

        logp_sum = np.sum(logps)
        nll = -logp_sum / sents
        ppl = np.exp(-logp_sum / words)
        return nll, ppl, logps
