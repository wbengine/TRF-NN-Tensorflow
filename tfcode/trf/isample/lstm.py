import tensorflow as tf
from copy import deepcopy
import numpy as np
import time
import json
import tqdm

from base import *
from lm import loss
from lm import stop


class Config(wb.Config):
    def __init__(self, data_or_vocab_size=1000, hidden_size=200, hidden_layers=2):
        if isinstance(data_or_vocab_size, int):
            self.vocab_size = data_or_vocab_size
        else:
            self.vocab_size = data_or_vocab_size.get_vocab_size()

        self.embedding_size = hidden_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.softmax_type = 'Softmax'  # only support 'Softmax'
        self.softmax_pred_token_per_cluster = int(np.sqrt(self.vocab_size))  # used to accelerate the sampling, suggested as int(np.sqrt(vocab_size))
        self.init_weight = 0.1
        self.optimize_method = 'SGD'  # can be one of 'SGD', 'adam'
        self.max_grad_norm = 5.0      # grad_norm, can be None
        self.dropout = 0.0
        self.learning_rate = 1.  # [learning rate] initial values
        self.max_update_batch = 100

    def __str__(self):
        s = 'lstm_e{}_h{}x{}'.format(self.embedding_size, self.hidden_size, self.hidden_layers)
        return s


class Net(object):
    def __init__(self, config, is_training, name='lstm_net', reuse=None):
        self.is_training = is_training
        self.config = config
        self.state = None  # save the current state of LSMT

        # a initializer for variables
        initializer = tf.random_uniform_initializer(-config.init_weight, config.init_weight)
        with tf.variable_scope(name, reuse=reuse, initializer=initializer):

            # Create LSTM cell
            def one_lstm_cell():
                c = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=0., reuse=reuse)
                if self.is_training and self.config.dropout > 0:
                    c = tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=1. - self.config.dropout_prob)
                return c

            cell = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.hidden_layers)])

            # inputs: [batch_size, max_length]
            # targets: [batch_size, max_length]
            # lengths: [batch_size]
            # _scales: [batch_size] the scales for each sequences
            self._inputs = tf.placeholder(tf.int32, [None, None], name='net_inputs')
            self._targets = tf.placeholder(tf.int32, [None, None], name='net_targets')
            self._lengths = tf.ones([tf.shape(self._inputs)[0]], dtype=tf.int32) * tf.shape(self._inputs)[1]
            self._scales = tf.placeholder(tf.float32, [None], name='net_scales')
            self._initial_state = cell.zero_state(tf.shape(self._inputs)[0], tf.float32)

            # bulid network...
            self.softmax, self.final_state = self.output(cell, self._initial_state,
                                                         self._inputs, self._targets, self._lengths)

            # loss for each position
            length_mask = tf.sequence_mask(self._lengths, tf.shape(self._inputs)[1], dtype=tf.float32)
            self.logps = self.softmax.logps
            self.logps_for_position = self.logps * length_mask
            self.logps_for_sequence = tf.reduce_sum(self.logps_for_position, axis=-1)
            self.loss = -self.softmax.logps
            self.loss_for_position = self.loss * length_mask
            self.loss_for_sequence = tf.reduce_sum(self.loss_for_position, axis=-1)

            # all the trainalbe vairalbes
            self.vars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
            self.var_size = tf.add_n([tf.size(v) for v in self.vars])
            # print the variables
            if reuse is None:
                print('[%s.%s] variables in %s' % (__name__, self.__class__.__name__, name))
                for v in self.vars:
                    print('\t' + v.name, v.shape, v.device)
                print('[%s.%s] max_update_batch=%d' % (__name__, self.__class__.__name__, self.config.max_update_batch))

            if is_training:
                # as _lr and _global_step is not defined in variables_scope(),
                # as the default collection is tf.GraphKeys.TRAINABLE_VARIABLES
                # It should be initialized by session.run(tf.global_variables_initializer()) or
                # use tf.Train.Supervisor()

                self.grads = tf.gradients(self.loss_for_sequence, self.vars, self._scales)
                self.train_op = layers.TrainOp(self.grads, self.vars,
                                               optimize_method=self.config.optimize_method,
                                               max_grad_norm=self.config.max_grad_norm)

    def output(self, cell, init_states, _inputs, _targets=None, _lengths=None):
        # word embedding
        word_embedding = tf.get_variable('word_embedding',
                                         [self.config.vocab_size, self.config.embedding_size],
                                         dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(word_embedding, _inputs)

        # dropout
        if self.is_training and self.config.dropout > 0:
            inputs = tf.nn.dropout(inputs, keep_prob=1. - self.config.dropout)

        # lstm
        # using tf.nn.dynamic_rnn to remove the constrains of step size.
        outputs, state = tf.nn.dynamic_rnn(cell,
                                           inputs=inputs,
                                           initial_state=init_states,
                                           sequence_length=_lengths)

        # softmax
        softmax = layers.Softmax(outputs, _targets, self.config.vocab_size,
                                 self.config.softmax_pred_token_per_cluster,
                                 name='Softmax')
        return softmax, state

    def run_update(self, session, inputs, targets, lengths, scales, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.config.learning_rate

        self.train_op.set_lr(session, learning_rate)
        self.train_op.update(session, {self._inputs: inputs,
                                       self._targets: targets,
                                       self._lengths: lengths,
                                       self._scales: scales})

    def run_loss(self, session, inputs, targets, lengths):
        return session.run(self.loss_for_sequence, {self._inputs: inputs,
                                                    self._targets: targets,
                                                    self._lengths: lengths})

    def run_draws(self, session, inputs, max_sample_num, end_id=None):
        states = session.run(self.final_state, {self._inputs: inputs[:, 0:-1]})

        final_outputs = np.array(inputs)
        final_lengths = np.ones(inputs.shape[0], dtype='int32') * inputs.shape[1]
        final_logps = np.zeros_like(inputs)
        is_tail = np.array([False] * len(inputs))
        for i in range(max_sample_num):
            draw_w, draw_logp, states = session.run(self.softmax.draw + [self.final_state],
                                                    {self._inputs: final_outputs[:, -1:],
                                                     self._initial_state: states})
            final_outputs = np.concatenate([final_outputs, draw_w], axis=-1)
            final_lengths += np.where(is_tail, 0, 1)
            final_logps = np.concatenate([final_logps, draw_logp], axis=-1)

            if end_id is not None:
                is_tail = np.logical_or(is_tail, np.reshape(draw_w, [-1]) == end_id)
                if np.all(is_tail):
                    break

        return final_outputs, final_lengths, final_logps

    def run_logps(self, session, inputs):
        final_logps = session.run(self.logps_for_position, {self._inputs: inputs[:, 0:-1],
                                                            self._targets: inputs[:, 1:]})
        final_logps = np.concatenate([np.zeros([len(inputs), 1]), final_logps], axis=-1)
        return final_logps


class Model(object):
    def __init__(self, config, name='seq_lstmlm', device='/gpu:0'):
        with tf.device(device):
            self.train_net = Net(config, is_training=True, name=name, reuse=None)
            self.eval_net = Net(config, is_training=False, name=name, reuse=True)

            self.saver = tf.train.Saver(self.train_net.vars)

    def update(self, session, seq_list, seq_scales, learning_rate=None):
        inputs, lengths = reader.produce_data_to_array(seq_list)
        self.train_net.run_update(session,
                                  inputs[:, 0:-1], inputs[:, 1:], lengths-1, seq_scales,
                                  learning_rate)

    def get_log_probs(self, session, seq_list, max_batch_size=100):
        logps = np.zeros(len(seq_list))
        for i in range(0, len(seq_list), max_batch_size):
            j = i + max_batch_size
            inputs, lengths = reader.produce_data_to_array(seq_list[i:j])
            logps[i: j] = -self.eval_net.run_loss(session, inputs[:, 0:-1], inputs[:, 1:], lengths-1)
        return logps

    def eval(self, session, seq_list):
        logps = self.get_log_probs(session, seq_list)
        nll = -np.mean(logps)
        words = np.sum([len(x) - 1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)

        return nll, ppl

    def draw_seqs(self, session, batch_size, beg_token, end_token=None, max_length=1000):
        inputs = np.ones([batch_size, 1], dtype='int32') * beg_token
        inputs, lengths, _ = self.eval_net.run_draws(session, inputs, max_length-1, end_token)

        if end_token is not None:
            inputs[np.arange(inputs.shape[0]), lengths-1] = end_token
        seq_list = reader.extract_data_from_array(inputs, lengths)
        return seq_list

    def add_noise(self, session, seq_list, sub=5):
        final_list = []

        inputs, _ = reader.produce_data_to_array(seq_list)
        for i, seq in enumerate(seq_list):
            pos = np.random.randint(1, len(seq)-1)
            outputs, lengths, _ = self.eval_net.run_draws(session, inputs[i:i+1, 0:pos], sub)
            if lengths[0] > len(seq)-1:
                out_seq = outputs[0][0: len(seq)]
                out_seq[-1] = seq[-1]
            else:
                out_seq = np.concatenate([outputs[0], seq[lengths[0]:]])

            final_list.append(out_seq.tolist())
        return final_list

    def save(self, session, file):
        self.saver.save(session, file)

    def restore(self, session, file):
        self.saver.restore(session, file)

