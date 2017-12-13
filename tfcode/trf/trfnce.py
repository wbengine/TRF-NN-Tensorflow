import tensorflow as tf
from copy import deepcopy
import numpy as np
import time
import json
import os
from collections import OrderedDict
from multiprocessing import Process, Manager, Queue, Value

from base import *
from lm import *
from trf import trfbase


class Config(wb.Config):
    def __init__(self, data):
        self.min_len = data.get_min_len()
        self.max_len = np.max(data.get_max_len())
        self.vocab_size = data.get_vocab_size()
        self.training_num = len(data.datas[0])
        self.valid_num = len(data.datas[1])
        self.pi_true = data.get_pi_true()
        self.beg_token = data.get_beg_token()
        self.end_token = data.get_end_token()
        self.global_normalized = False
        self.structure_type = 'cnn'  # 'cnn' or 'rnn' or 'mix'
        self.loss = 'nce'  # nce or softmax
        self.reference_model = None  # such as 'lstm:<PATH>'
        # cnn structure
        self.embedding_dim = 128
        self.load_embedding_path = None
        self.cnn_filters = [(i, 128) for i in range(1, 6)]
        self.cnn_layers = 3
        self.cnn_hidden = 128
        self.cnn_width = 3
        self.cnn_activation = 'relu'
        self.cnn_batch_normalize = False
        self.cnn_skip_connection = True
        self.cnn_residual = False
        # lstm structure
        self.rnn_hidden_size = 200
        self.rnn_hidden_layers = 2
        self.rnn_type = 'blstm'  # 'lstm' or 'rnn' or 'blstm', 'brnn'
        self.rnn_predict = False
        self.attention = False
        # for learning
        self.dropout = 0
        self.noise_operation_num = 5
        self.optimize_method = ['adam', 'adam']
        self.max_grad_norm = [10, 10]
        self.batch_size = 100
        self.noise_factor = 10
        self.noise_sampler = '2gram'  # str, such as 1gram, 2gram or lstm
        self.noise_multiple_use = 1
        self.max_epoch = 100
        self.lr_param = trfbase.LearningRateEpochDelay(1e-3)
        self.lr_zeta = trfbase.LearningRateEpochDelay(1e-2)
        self.init_weight = 0.1
        self.init_zeta = self.get_initial_logz()
        self.update_zeta = False
        # for debug
        self.write_dbg = True

    def get_initial_logz(self, c=None):
        if c is None:
            c = np.log(self.vocab_size)
        len_num = self.max_len - self.min_len + 1
        logz = c * (1 + np.linspace(1, len_num, len_num))
        return logz

    def __str__(self):
        if self.global_normalized:
            s = 'grf'
        else:
            s = 'trf'

        if self.cnn_batch_normalize:
            s += '_BN'

        s += '_%s%d' % (self.loss, self.noise_factor)

        s += '_e{}'.format(self.embedding_dim)
        # cnn structure
        if self.structure_type == 'cnn' or self.structure_type == 'mix':
            s += '_cnn'
            if self.cnn_filters:
                a = list(map(lambda x: x[1] == self.cnn_filters[0][1], self.cnn_filters))
                if all(a):
                    s += '_({}to{})x{}'.format(self.cnn_filters[0][0],
                                               self.cnn_filters[-1][0],
                                               self.cnn_filters[0][1])
                else:
                    s += '_' + ''.join(['({}x{})'.format(w, d) for (w, d) in self.cnn_filters])
            if self.cnn_layers > 0:
                s += '_({}x{})x{}'.format(self.cnn_width, self.cnn_hidden, self.cnn_layers)
            s += '_{}'.format(self.cnn_activation)

        # rnn structure
        if self.structure_type[0:3] == 'rnn' or self.structure_type == 'mix':
            s += '_{}_{}x{}'.format(self.rnn_type, self.rnn_hidden_size, self.rnn_hidden_layers)
            if self.rnn_predict:
                s += '_pred'

        if self.attention:
            s += '_at'

        if self.noise_sampler is not None:
            s += '_noise%s' % self.noise_sampler.split(':')[0]
        if self.update_zeta:
            s += '_updatezeta'

        if self.reference_model is not None:
            s += '_with_%s' % self.reference_model.split('/')[-2]
        return s


class NetBase(object):
    def __init__(self, config, is_training, device='/gpu:0', name='net', reuse=None):
        self.is_training = is_training
        self.config = config

        # if training, the batch_size is fixed, make the training more effective
        # batch_size = self.config.batch_size * (1 + self.config.noise_factor) if is_training else None
        batch_size = None

        default_initializer = tf.random_uniform_initializer(-config.init_weight, config.init_weight)
        with tf.device(device), tf.variable_scope(name, reuse=reuse, initializer=default_initializer):
            #############################################
            # inputs: of shape (batch_size, seq_len)
            # lengths: of shape (batch_size,)
            #############################################
            self._inputs = tf.placeholder(tf.int32, [batch_size, None], name='inputs')
            self._lengths = tf.placeholder(tf.int32, [batch_size], name='lengths')
            self._q_logps = tf.placeholder(tf.float32, [batch_size], name='q_logps')
            #############################################################
            # compute the energy function phi
            #############################################################
            self.phi, self.vars = self.output(config, self._inputs, self._lengths, reuse=reuse)
            self.var_size = tf.add_n([tf.size(v) for v in self.vars])
            # print the variables
            if reuse is None:
                print('variables in %s' % name)
                for v in self.vars:
                    print('\t' + v.name, v.shape, v.device)

            #############################################################
            # compute the log probs
            #############################################################
            if config.global_normalized:
                self.zeta = tf.get_variable('global_zeta', shape=[1], dtype=tf.float32,
                                            initializer=tf.constant_initializer([config.init_zeta]))
                self.logps = self.phi - self.zeta[0] + self._q_logps
            else:
                valid_len = config.max_len - config.min_len + 1
                self.pi = tf.get_variable('pi', shape=[valid_len], dtype=tf.float32,
                                          trainable=False,
                                          initializer=tf.constant_initializer(config.pi_true[config.min_len:]))
                self.zeta = tf.get_variable('zeta', shape=[valid_len], dtype=tf.float32,
                                            initializer=tf.constant_initializer(config.init_zeta))

                norm_constant = tf.log(self.pi) - self.zeta
                self.logps = self.phi + tf.gather(norm_constant, self._lengths - config.min_len) + self._q_logps
            # set zeta
            self._new_zeta = tf.placeholder(tf.float32, [None], name='new_logz')
            self._update_zeta = tf.assign(self.zeta, self._new_zeta)

            # set zeta1
            self._new_zeta_base = tf.placeholder(tf.float32, [], name='logz_base')
            self._update_zeta_base = tf.assign_add(self.zeta,
                                                   tf.ones_like(self.zeta) * (-self.zeta[0] + self._new_zeta_base)
                                                   )

            ###################################################
            # for NCE training
            ###################################################
            if is_training:
                #############################################################
                # compute the loss
                #############################################################
                # the noise log-probs for all the input sequences, of shape [batch_size]
                self._noise_logps = tf.placeholder(tf.float32, shape=[batch_size], name='noise_logps')
                self._data_num = tf.placeholder(tf.int32, shape=[], name='default_data_num')

                # compute the loss
                if config.loss == 'nce':
                    print('[Net] NCE loss')
                    self.loss, \
                    self.grad_for_params,\
                    self.grad_for_zeta, \
                    self.dbg = self.nce_loss(self._noise_logps,  self._data_num)
                else:
                    print('[Net] softmax loss')
                    self.loss, \
                    self.grad_for_params,\
                    self.grad_for_zeta, \
                    self.dbg = self.softmax_loss(self._noise_logps,  self._data_num)

                # self.grads_list = [tf.gradients(self.loss, self.vars), tf.gradients(self.loss, self.zeta)]
                self.grads_list = [self.grad_for_params, self.grad_for_zeta]

                ###################################
                # training
                ###################################
                self._lr = [tf.Variable(1.0, trainable=False, name='learning_rate_param'),
                            tf.Variable(1.0, trainable=False, name='learning_rate_zeta')]
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                for i, tvars in enumerate([self.vars, [self.zeta]]):
                    grads = self.grads_list[i]
                    # compute the gradient
                    if config.max_grad_norm[i] is not None:
                        grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm[i])
                    # optimizer
                    if config.optimize_method[i].lower() == 'adam':
                        optimizer = tf.train.AdamOptimizer(self._lr[i])
                    elif config.optimize_method[i].lower() == 'adagrad':
                        optimizer = tf.train.AdagradOptimizer(self._lr[i])
                    else:
                        optimizer = tf.train.GradientDescentOptimizer(self._lr[i])

                    # update parameters
                    if i == 0:
                        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
                    else:
                        self.train_zeta = optimizer.apply_gradients(zip(grads, tvars))

                # set learning rate
                self._new_lr = tf.placeholder(tf.float32, [])
                self._set_lr = [tf.assign(self._lr[i], self._new_lr) for i in range(len(self._lr))]

    def nce_loss(self, noise_logps, data_num):
        # data_number
        noise_num = tf.shape(self._inputs)[0] - data_num
        data_num_float = tf.cast(data_num, tf.float32)
        # word_num_float = tf.cast(tf.reduce_sum(self._lengths - 1), tf.float32)

        # cluster probability p(C=0|x)
        diff_logps = noise_logps - self.logps

        temp_stack = tf.stack([tf.zeros_like(diff_logps),
                               np.log(self.config.noise_factor) + diff_logps])
        log_cluster_prob = -tf.reduce_logsumexp(temp_stack, axis=0)

        data_loss = tf.reduce_sum(log_cluster_prob[0: data_num])
        noise_loss = tf.reduce_sum(np.log(self.config.noise_factor) + diff_logps[data_num:] +
                                   log_cluster_prob[data_num:])
        loss = -(data_loss + noise_loss) / data_num_float

        data_p = tf.exp(log_cluster_prob[0: data_num])
        noise_p = tf.exp(log_cluster_prob[data_num:])

        ###############################
        # avoid the over-fitting caused by the diff_lops too large or too small
        ###############################
        # gradient for params
        grad_loss_for_phi = tf.concat([data_p-1, noise_p], axis=0) / data_num_float
        grad_loss_for_phi_at_inf = tf.concat([-tf.ones([data_num], dtype=tf.float32),
                                              tf.zeros([noise_num], dtype=tf.float32)],
                                             axis=0) / data_num_float
        # grad_loss_for_phi_at_zero = tf.concat([tf.zeros([data_num], dtype=tf.float32),
        #                                        tf.ones([noise_num], dtype=tf.float32)],
        #                                       axis=0) / data_num_float
        # grad_loss_for_phi = tf.reshape(grad_loss_for_phi, [-1])
        grad_loss_for_phi = tf.where(tf.less(log_cluster_prob, -40),
                                     grad_loss_for_phi_at_inf,
                                     grad_loss_for_phi)
        # grad_loss_for_phi = tf.where(tf.less(diff_logps, -40),
        #                              grad_loss_for_phi_at_zero,
        #                              grad_loss_for_phi)

        # self.grad_for_params = tf.gradients(self.phi, self.vars, tf.reshape(grad_loss_for_phi, [1, -1]))
        grad_for_params = tf.gradients(self.phi, self.vars, grad_loss_for_phi)

        # gradient for zeta
        grad_for_zeta = tf.scatter_nd(tf.reshape(self._lengths - self.config.min_len, [-1, 1]),
                                      -grad_loss_for_phi,
                                      shape=self.zeta.shape)
        grad_for_zeta = [grad_for_zeta]

        dbgs = {
            'loss': loss,
            'logpn-logpm': diff_logps,
            'P(C)': tf.exp(log_cluster_prob),
            'phi': self.phi,
            'logps': self.logps,
            'dphi': grad_loss_for_phi,
            'dzeta': grad_for_zeta[0],
            'lengths': self._lengths}

        return loss, grad_for_params, grad_for_zeta, dbgs

    def softmax_loss(self, noise_logps, data_num):

        log_cluster_prob = self.logps - tf.reduce_logsumexp(self.logps)
        loss = -tf.reduce_mean(log_cluster_prob[0: data_num])

        grad_for_params = tf.gradients(loss, self.vars)

        delta_for_per_zeta_l = []
        delta_for_all_zeta = self.phi - noise_logps
        for i in range(self.config.max_len - self.config.min_len + 1):
            w = tf.dynamic_partition(delta_for_all_zeta[data_num:],
                                     tf.cast(tf.equal(self._lengths[data_num], i+self.config.min_len), dtype=tf.int32),
                                     2)[-1]
            d = tf.reduce_logsumexp(w)
            delta_for_per_zeta_l.append(tf.where(tf.is_inf(d), self.zeta[i], d))

        grad_for_zeta = [-tf.stack(delta_for_per_zeta_l) + self.zeta]

        dbgs = {
            'loss': loss,
            'P(C)': tf.exp(log_cluster_prob),
            'phi': self.phi,
            'logps': self.logps,
            'dzeta': grad_for_zeta[0],
            'lengths': self._lengths}

        return loss, grad_for_params, grad_for_zeta, dbgs

    def output(self, config, _inputs, _lengths, reuse=None):
        pass

    def get_logz(self, session):
        return session.run(self.zeta)

    def set_logz(self, session, logz):
        return session.run(self._update_zeta, {self._new_zeta: logz})

    def set_logz_base(self, session, base):
        return session.run(self._update_zeta_base, {self._new_zeta_base: base})

    def run_phi(self, session, inputs, lengths):
        return session.run(self.phi,
                           {self._inputs: inputs,
                            self._lengths: lengths})

    def run_logps(self, session, inputs, lengths, qlogps):
        return session.run(self.logps,
                           {self._inputs: inputs,
                            self._lengths: lengths,
                            self._q_logps: qlogps})

    def run(self, session, inputs, lengths, qlogps, noise_logps, ops=None):
        return session.run(ops,
                           {self._inputs: inputs,
                            self._lengths: lengths,
                            self._data_num: self.config.batch_size,
                            self._q_logps: qlogps,
                            self._noise_logps: noise_logps})

    def run_train(self, session, inputs, lengths, qlogps, noise_logps):
        return self.run(session, inputs, lengths, qlogps, noise_logps, [self.train_op, self.train_zeta])
        # return self.run(session, inputs, lengths, noise_logps, self.train_op)

    def run_set_lr(self, session, lr1, lr2):
        session.run(self._set_lr[0], {self._new_lr: lr1})
        session.run(self._set_lr[1], {self._new_lr: lr2})


class NetCNN(NetBase):
    def activation(self, x, reuse, name='BN', add_activation=True):
        if self.config.cnn_batch_normalize:
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                             is_training=self.is_training,
                                             scope=name, reuse=reuse)

        if not add_activation:
            return x

        if self.config.cnn_activation is None:
            x = x
        elif self.config.cnn_activation == 'relu':
            x = tf.nn.relu(x)
        elif self.config.cnn_activation == 'tanh':
            x = tf.nn.tanh(x)
        elif self.config.cnn_activation == 'sigmod':
            x = tf.nn.sigmoid(x)
        else:
            raise TypeError('unknown activation {}'.format(self.config.cnn_activation))

        return x

    def output(self, config, _inputs, _lengths, reuse=None):
        """
        Using the self._inputs and self._lengths to calculate phi of TRF
        """
        vocab_size = config.vocab_size
        embedding_dim = config.embedding_dim

        batch_size = tf.shape(_inputs)[0]
        max_len = tf.shape(_inputs)[1]
        # the length mask, the position < len is 1., otherwise is 0.
        len_mask = tf.tile(tf.reshape(tf.range(max_len, dtype=tf.int32), [1, max_len]), [batch_size, 1])
        len_mask = tf.less(len_mask, tf.reshape(_lengths, [batch_size, 1]))
        len_mask = tf.cast(len_mask, tf.float32)  # shape (batch_size, max_len)
        expand_len_mask = tf.expand_dims(len_mask, axis=-1)  # shape (batch_size, max_len, 1)
        tf.summary.image('len_mask', tf.expand_dims(expand_len_mask, axis=0), collections=['cnn'])

        # embedding layers
        if config.load_embedding_path is not None:
            print('read init embedding vector from', config.load_embedding_path)
            emb_init_value = word2vec.read_vec(config.load_embedding_path)
            if emb_init_value.shape != (vocab_size, embedding_dim):
                raise TypeError('the reading embedding with shape ' +
                                str(emb_init_value.shape) +
                                ' does not match current shape ' +
                                str([vocab_size, embedding_dim]) +
                                '\nform path ' + config.load_embedding_path)
            word_embedding = tf.get_variable('word_embedding',
                                             [vocab_size, embedding_dim], dtype=tf.float32,
                                             initializer=tf.constant_initializer(emb_init_value),
                                             trainable=True)
        else:
            word_embedding = tf.get_variable('word_embedding',
                                             [vocab_size, embedding_dim], dtype=tf.float32)
        emb_output = tf.nn.embedding_lookup(word_embedding, _inputs)  # (batch_size, seq_len, emb_dim)

        # dropout
        if self.is_training and self.config.dropout > 0:
            emb_output = tf.nn.dropout(emb_output, keep_prob=1. - self.config.dropout)

        # pre-net
        emb_output = layers.linear(emb_output, config.cnn_hidden, tf.nn.relu, name='pre_net1')
        emb_output = emb_output * expand_len_mask
        # dropout
        if self.is_training and self.config.dropout > 0:
            emb_output = tf.nn.dropout(emb_output, keep_prob=1. - self.config.dropout)

        inputs = emb_output
        tf.summary.image('embedding', tf.expand_dims(inputs, axis=-1), max_outputs=4, collections=['cnn'])
        tf.summary.histogram('embedding', word_embedding, collections=['cnn'])

        # cnn layer-0
        # compute cnn with different filter width
        conv_list = []
        for (filter_width, out_dim) in config.cnn_filters:
            # cnn output is of shape (bacth_size, seq_len, out_dim)
            conv = tf.layers.conv1d(
                inputs=inputs,  # set the values at positon >= length to zeros
                filters=out_dim,
                kernel_size=filter_width,
                padding='same',
                activation=None,
                reuse=reuse,
                name='cnn0_{}'.format(filter_width)
            )
            conv = self.activation(conv, reuse, 'cnn0_{}/BN'.format(filter_width))
            conv_list.append(conv * expand_len_mask)

        if conv_list:
            inputs = tf.concat(conv_list, axis=-1)
        tf.summary.image('cnn0', tf.expand_dims(inputs, axis=-1), max_outputs=4, collections=['cnn'])

        # max_pooling
        inputs = tf.layers.max_pooling1d(inputs, pool_size=2, strides=1, padding='same')
        inputs *= expand_len_mask
        tf.summary.image('cnn0_pooling', tf.expand_dims(inputs, axis=-1), max_outputs=4, collections=['cnn'])

        # several cnn layers
        skip_connections = []
        for i in range(config.cnn_layers):
            conv_output = tf.layers.conv1d(
                inputs=inputs,
                filters=config.cnn_hidden,
                kernel_size=config.cnn_width,
                padding='same',
                activation=None,
                reuse=reuse,
                name='cnn{}'.format(i + 1)
            )
            conv_output = self.activation(conv_output, reuse, 'cnn{}/BN'.format(i+1))
            conv_output = conv_output * expand_len_mask
            tf.summary.image('cnn{}'.format(i + 1),
                             tf.expand_dims(conv_output, axis=-1),
                             max_outputs=4, collections=['cnn'])

            if config.cnn_skip_connection:
                skip_scalar = tf.get_variable(name='cnn{}_skip_scalar'.format(i + 1),
                                              shape=[config.cnn_hidden], dtype=tf.float32)
                skip_connections.append(conv_output * tf.reshape(skip_scalar, [1, 1, config.cnn_hidden]))

            inputs = conv_output

        # skip connections
        if skip_connections:
            inputs = self.activation(tf.add_n(skip_connections), reuse, 'skip_conn/BN')

        # residual connection
        if config.cnn_residual:
            inputs = tf.nn.relu(emb_output + inputs)

        tf.summary.image('cnn_end',
                         tf.expand_dims(inputs, axis=-1),
                         max_outputs=4, collections=['cnn'])

        ### TRF outputs
        # final conv
        conv_output = tf.layers.conv1d(
            inputs=inputs,
            filters=1,
            kernel_size=1,
            padding='valid',
            activation=None,
            use_bias=True,
            reuse=reuse,
            name='cnn_final'
        )
        outputs = tf.reshape(conv_output, [batch_size, -1])

        outputs = outputs * len_mask
        outputs = tf.reduce_sum(outputs, axis=-1)  # of shape [batch_size]

        return outputs, tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)


class NetRnn(NetBase):
    def compute_rnn(self, _inputs, _lengths, reuse=True):
        # LSTM Cells
        # lstm cell
        # Create LSTM cell
        def one_lstm_cell():
            if self.config.rnn_type.find('lstm') != -1:
                c = tf.contrib.rnn.BasicLSTMCell(self.config.rnn_hidden_size, forget_bias=0., reuse=reuse)
            elif self.config.rnn_type.find('rnn') != -1:
                c = tf.contrib.rnn.BasicRNNCell(self.config.rnn_hidden_size, activation=tf.nn.tanh, reuse=reuse)
            else:
                raise TypeError('undefined rnn type = ' + self.config.type)
            if self.is_training and self.config.dropout > 0:
                c = tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=1. - self.config.dropout)
            return c


        # vocab_size = self.config.vocab_size
        # embedding_dim = self.config.embedding_dim

        batch_size = tf.shape(_inputs)[0]

        # embedding layers
        word_embedding = tf.get_variable('word_embedding',
                                         [self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32)
        emb = tf.nn.embedding_lookup(word_embedding, _inputs)  # (batch_size, seq_len, emb_dim)
        inputs = emb

        # dropout
        if self.is_training and self.config.dropout > 0:
            inputs = tf.nn.dropout(inputs, keep_prob=1. - self.config.dropout)

        # recurrent structure
        if self.config.rnn_type[0].lower() == 'b':
            cell_fw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.rnn_hidden_layers)])
            cell_bw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.rnn_hidden_layers)])
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                              inputs=inputs,
                                                              sequence_length=_lengths,
                                                              dtype=tf.float32)
            outputs_fw = outputs[0]
            outputs_bw = outputs[1]

        else:
            cell_fw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.rnn_hidden_layers)])
            outputs, states = tf.nn.dynamic_rnn(cell_fw, inputs=inputs, sequence_length=_lengths, dtype=tf.float32)
            outputs_fw = outputs
            outputs_bw = None

        return outputs_fw, outputs_bw, states, emb

    def output(self, config, _inputs, _lengths, reuse=None):

        outputs_fw, outputs_bw, _, emb = self.compute_rnn(_inputs, _lengths, reuse)

        batch_size = tf.shape(_inputs)[0]

        if not config.rnn_predict:
            inputs = tf.concat([outputs_fw, outputs_bw], axis=2)

            # final layers
            batch_size = tf.shape(_inputs)[0]
            len_mask = tf.sequence_mask(_lengths, maxlen=tf.shape(_inputs)[1], dtype=tf.float32)
            outputs = layers.linear(inputs, 1, name='final_linear')  # [batch_size, max_len, 1]
            outputs = tf.reshape(outputs, [batch_size, -1])  # [batch_size, max_len]
            outputs = outputs * len_mask
            outputs = tf.reduce_sum(outputs, axis=-1)  # of shape [batch_size]

        else:

            softmax_w = tf.get_variable('final_pred/w',
                                        [self.config.rnn_hidden_size, self.config.vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable('final_pred/b', [self.config.vocab_size], dtype=tf.float32)

            def final_pred(inputs, labels):
                outputs = tf.reshape(inputs, [-1, self.config.rnn_hidden_size])  # [batch_size*seq_len, hidden_size]
                logits = tf.matmul(outputs, softmax_w) + softmax_b  # [batch_size*seq_len, vocab_size]

                labels = tf.reshape(labels, [-1])
                idx = tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1)
                loss = tf.gather_nd(logits, idx)  # [batch_size * seq_len]

                loss = tf.reshape(loss, [batch_size, -1])  # [batch_size, seq_len]
                return loss

            # outputs_fw = tf.reduce_sum(outputs_fw[:, 0:-1] * emb[:, 1:], axis=-1)
            # outputs_bw = tf.reduce_sum(outputs_bw[:, 1:] * emb[:, 0:-1], axis=-1)
            outputs = 0
            if outputs_fw is not None:
                outputs += final_pred(outputs_fw[:, 0:-1], _inputs[:, 1:])
            if outputs_bw is not None:
                outputs += final_pred(outputs_bw[:, 1:], _inputs[:, 0:-1])

            outputs *= tf.sequence_mask(_lengths-1, maxlen=tf.shape(_inputs)[1]-1, dtype=tf.float32)
            outputs = tf.reduce_sum(outputs, axis=-1)  # of shape [batch_size]

        return outputs, tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)


class NetMix(NetBase):
    def compute_rnn(self, inputs, _lengths, reuse=True):
        # LSTM Cells
        # lstm cell
        # Create LSTM cell
        def one_lstm_cell():
            if self.config.rnn_type.find('lstm') != -1:
                c = tf.contrib.rnn.BasicLSTMCell(self.config.rnn_hidden_size, forget_bias=0., reuse=reuse)
            elif self.config.rnn_type.find('rnn') != -1:
                c = tf.contrib.rnn.BasicRNNCell(self.config.rnn_hidden_size, activation=tf.nn.tanh, reuse=reuse)
            else:
                raise TypeError('undefined rnn type = ' + self.config.rnn_type)
            if self.is_training and self.config.dropout > 0:
                c = tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=1. - self.config.dropout)
            return c

        cell_fw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.rnn_hidden_layers)])
        cell_bw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.rnn_hidden_layers)])

        batch_size = tf.shape(inputs)[0]

        # dropout
        if self.is_training and self.config.dropout > 0:
            inputs = tf.nn.dropout(inputs, keep_prob=1. - self.config.dropout)

        # lstm
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                          inputs=inputs,
                                                          sequence_length=_lengths,
                                                          initial_state_fw=cell_fw.zero_state(batch_size, tf.float32),
                                                          initial_state_bw=cell_fw.zero_state(batch_size, tf.float32))

        return outputs[0], outputs[1], states

    def activation(self, x, reuse, name='BN', add_activation=True):
        if self.config.cnn_batch_normalize:
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                             is_training=self.is_training,
                                             scope=name, reuse=reuse)

        if not add_activation:
            return x

        if self.config.cnn_activation is None:
            x = x
        elif self.config.cnn_activation == 'relu':
            x = tf.nn.relu(x)
        elif self.config.cnn_activation == 'tanh':
            x = tf.nn.tanh(x)
        elif self.config.cnn_activation == 'sigmod':
            x = tf.nn.sigmoid(x)
        else:
            raise TypeError('unknown activation {}'.format(self.config.cnn_activation))

        return x

    def output(self, config, _inputs, _lengths, reuse=None):
        """
        Using the self._inputs and self._lengths to calculate phi of TRF
        """
        vocab_size = config.vocab_size
        embedding_dim = config.embedding_dim

        batch_size = tf.shape(_inputs)[0]
        max_len = tf.shape(_inputs)[1]
        # the length mask, the position < len is 1., otherwise is 0.
        len_mask = tf.sequence_mask(_lengths, max_len, dtype=tf.float32)  # shape (batch_size, max_len)
        expand_len_mask = tf.expand_dims(len_mask, axis=-1)  # shape (batch_size, max_len, 1)

        # embedding layers
        word_embedding = tf.get_variable('word_embedding', [vocab_size, embedding_dim], dtype=tf.float32)
        emb_output = tf.nn.embedding_lookup(word_embedding, _inputs)  # (batch_size, seq_len, emb_dim)

        # dropout
        if self.is_training and self.config.dropout > 0:
            emb_output = tf.nn.dropout(emb_output, keep_prob=1. - self.config.dropout)

        # cnn layer-0
        # compute cnn with different filter width
        inputs = emb_output
        conv_list = []
        for (filter_width, out_dim) in config.cnn_filters:
            # cnn output is of shape (bacth_size, seq_len, out_dim)
            conv = tf.layers.conv1d(
                inputs=inputs,  # set the values at positon >= length to zeros
                filters=out_dim,
                kernel_size=filter_width,
                padding='same',
                activation=None,
                reuse=reuse,
                name='cnn0_{}'.format(filter_width)
            )
            conv = self.activation(conv, reuse, 'cnn0_{}/BN'.format(filter_width))
            conv_list.append(conv * expand_len_mask)

        if conv_list:
            inputs = tf.concat(conv_list, axis=-1)

        # max_pooling
        # inputs = tf.layers.max_pooling1d(inputs, pool_size=2, strides=1, padding='same')
        # inputs *= expand_len_mask
        # tf.summary.image('cnn0_pooling', tf.expand_dims(inputs, axis=-1), max_outputs=4, collections=['cnn'])

        # several cnn layers
        skip_connections = []
        for i in range(config.cnn_layers):
            conv_output = tf.layers.conv1d(
                inputs=inputs,
                filters=config.cnn_hidden,
                kernel_size=config.cnn_width,
                padding='same',
                activation=None,
                reuse=reuse,
                name='cnn{}'.format(i + 1)
            )
            conv_output = self.activation(conv_output, reuse, 'cnn{}/BN'.format(i+1))
            conv_output = conv_output * expand_len_mask

            if config.cnn_skip_connection:
                skip_scalar = tf.get_variable(name='cnn{}_skip_scalar'.format(i + 1),
                                              shape=[config.cnn_hidden], dtype=tf.float32)
                skip_connections.append(conv_output * tf.reshape(skip_scalar, [1, 1, config.cnn_hidden]))

            inputs = conv_output

        # skip connections
        if skip_connections:
            inputs = self.activation(tf.add_n(skip_connections), reuse, 'skip_conn/BN')

        if config.cnn_residual:
            inputs += emb_output

        #############################
        ##### LSTM part #############
        #############################
        outputs_fw, outputs_bw, _, = self.compute_rnn(inputs, _lengths, reuse)
        inputs = tf.concat([outputs_fw, outputs_bw], axis=2)

        if config.attention:
            attention_weight = layers.linear(inputs, 1, activate=tf.nn.sigmoid, name='attention_weight')
            # summate
            inputs *= attention_weight

        # final layers
        outputs = tf.reduce_sum(inputs * expand_len_mask, axis=1)  # [batch_size, dim]
        outputs = layers.linear(outputs, 1, name='final_linear')  # [batch_size, 1]
        outputs = tf.squeeze(outputs, axis=[-1])  # [batch_size]

        return outputs, tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)


class DistributedNet(object):
    """
    a distributed warper for network
    """
    def __init__(self, network, config, is_training,
                 device=['/gpu:0'], name='trf', reuse=None):

        self.distributed_num = len(device)
        self.config = config

        with tf.device(device[0]), tf.name_scope('main_device'):
            self.net_list = []
            for i, dev in enumerate(device):
                with tf.name_scope('device_%d' % i):
                    self.net_list.append(
                        network(config, is_training=is_training,
                                device=dev, name=name,
                                reuse=reuse if i == 0 else True)
                    )

            # all the phi
            self.phi = tf.concat([net.phi for net in self.net_list], axis=0)
            self.logps = tf.concat([net.logps for net in self.net_list], axis=0)

            self.vars = self.net_list[0].vars
            self.var_size = self.net_list[0].var_size
            self.zeta = self.net_list[0].zeta

            if is_training:
                # total cost
                self.loss = tf.add_n([net.loss for net in self.net_list]) / len(self.net_list)
                # total dbg
                self.dbg = self.net_list[0].dbg

                # average all the gradients
                def average_distributed_grad(grads_list):
                    avg_grads = []
                    for gs in zip(*grads_list):
                        # gs is like (g0_at_gpu0, g0_at_gpu1, ..., g0_at_gpuN), ...
                        grads_map = tf.concat([tf.expand_dims(g, axis=0) for g in gs], axis=0)
                        avg_grads.append(tf.reduce_mean(grads_map, axis=0))
                    return avg_grads

                # average gradients
                avg_grad_vars = average_distributed_grad([net.grad_for_params for net in self.net_list])
                avg_grad_zeta = average_distributed_grad([net.grad_for_zeta for net in self.net_list])
                self.grads_list = [avg_grad_vars, avg_grad_zeta]

                # learining rate
                self._lr = [tf.Variable(1.0, trainable=False, name='learning_rate_param'),
                            tf.Variable(1.0, trainable=False, name='learning_rate_zeta')]
                # global step
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                # training operations
                for i, tvars in enumerate([self.vars, [self.zeta]]):
                    grads = self.grads_list[i]
                    lr = self._lr[i]
                    # clip gradient
                    if config.max_grad_norm[i] is not None:
                        grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm[i])
                    # optimizer
                    if config.optimize_method[i].lower() == 'adam':
                        optimizer = tf.train.AdamOptimizer(lr)
                    elif config.optimize_method[i].lower() == 'adagrad':
                        optimizer = tf.train.AdagradOptimizer(lr)
                    else:
                        optimizer = tf.train.GradientDescentOptimizer(lr)

                    # update parameters
                    if i == 0:
                        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
                    else:
                        self.train_zeta = optimizer.apply_gradients(zip(grads, tvars))

                # set learning rate
                self._new_lr = tf.placeholder(tf.float32, [])
                self._set_lr = [tf.assign(self._lr[i], self._new_lr) for i in range(len(self._lr))]

    def get_logz(self, session):
        return session.run(self.zeta)

    def set_logz(self, session, logz):
        return self.net_list[0].set_logz(session, logz)

    def run_infer(self, session, inputs, lengths, ops):
        n = len(inputs)
        if n < self.distributed_num:
            inputs = np.concatenate([inputs, np.repeat(inputs[-1:], self.distributed_num-n, axis=0)], axis=0)
            lengths = np.concatenate([lengths, np.repeat(lengths[-1:], self.distributed_num)])
            assert len(inputs) == self.distributed_num

        feed_dict = dict()
        num_per_device = int(np.ceil(len(inputs) / self.distributed_num))
        for i in range(self.distributed_num):
            beg = i * num_per_device
            end = beg + num_per_device
            feed_dict[self.net_list[i]._inputs] = inputs[beg: end]
            feed_dict[self.net_list[i]._lengths] = lengths[beg: end]
        return session.run(ops, feed_dict)[0:n]

    def run_phi(self, session, inputs, lengths):
        return self.run_infer(session, inputs, lengths, self.phi)

    def run_logps(self, session, inputs, lengths):
        return self.run_infer(session, inputs, lengths, self.logps)

    def run(self, session, inputs, lengths, noise_logps, ops=None):
        data_num = self.config.batch_size
        noise_num = len(inputs) - data_num

        if data_num % self.distributed_num != 0 or noise_num % self.distributed_num != 0:
            raise TypeError('[DistributedNet] run: the data_num [%d] or noise_num [%d] is not valid, '
                            'for distributed num [%d]' % (data_num, noise_num, self.distributed_num))

        data_num_per_device = data_num // self.distributed_num
        noise_num_per_device = noise_num // self.distributed_num
        feed_dict = dict()
        for i in range(self.distributed_num):

            # extract data
            data_beg = data_num_per_device * i
            data_end = data_num_per_device * (i + 1)

            noise_beg = data_num + noise_num_per_device * i
            noise_end = data_num + noise_num_per_device * (i + 1)

            merge_inputs = np.concatenate([inputs[data_beg: data_end], inputs[noise_beg: noise_end]], axis=0)
            merge_lengths = np.concatenate([lengths[data_beg: data_end], lengths[noise_beg: noise_end]], axis=0)
            merge_nlogps = np.concatenate([noise_logps[data_beg: data_end], noise_logps[noise_beg: noise_end]], axis=0)

            feed_dict[self.net_list[i]._inputs] = merge_inputs
            feed_dict[self.net_list[i]._lengths] = merge_lengths
            feed_dict[self.net_list[i]._data_num] = data_num_per_device
            feed_dict[self.net_list[i]._noise_logps] = merge_nlogps
        return session.run(ops, feed_dict)

    def run_train(self, session, inputs_list, lengths_list, noise_logps_list):
        return self.run(session, inputs_list, lengths_list,
                        noise_logps_list,
                        [self.train_op, self.train_zeta])

        # return self.run(session, inputs_list, lengths_list,
        #                 noise_logps_list,
        #                 ops=self.train_op)

    def run_set_lr(self, session, lr1, lr2):
        session.run(self._set_lr[0], {self._new_lr: lr1})
        session.run(self._set_lr[1], {self._new_lr: lr2})


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
                                                self.config.batch_size * self.config.noise_factor))

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
            seqs = self.noise_generate(self.config.batch_size * self.config.noise_factor)
            return seqs, self.noise_logps(seqs)

    def start(self):
        # prepare for the sampling
        if self.is_parallel:
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
        self.unigram[config.beg_token] = 0
        self.unigram[config.end_token] = 0
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
            rand_s[-1] = self.config.end_token
            seqs.append(rand_s)

        logps = self.noise_logps(seqs)
        return seqs, logps

    def noise_logps(self, seq_list):
        logps = []
        for seq in seq_list:
            a = [self.unigram[i] for i in seq[1:-1]]
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
            for _ in range(rand_len-2):
                p = self.ngram.get_prob(rand_s)
                w = p.sample()
                rand_s.append(w)

            rand_s.append(self.config.end_token)
            seqs.append(rand_s)
        return seqs, self.noise_logps(seqs)

    def noise_logps(self, seq_list):
        logps = []
        for seq in seq_list:
            a = []
            for i in range(1, len(seq)-1):
                p = self.ngram.get_prob(seq[0:i])
                a.append(p[seq[i]])
            logps.append(np.sum(np.log(a)) + np.log(self.config.pi_true[len(seq)]))

        return np.array(logps)


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


class TRF(object):
    def __init__(self, config, data,
                 name='trf', logdir='trf',
                 device='/gpu:0'):

        self.config = config
        self.data = data
        self.logdir = logdir
        self.name = name
        self.session = None

        network = self.parser_net(config.structure_type)

        if isinstance(device, list):
            self.distributed_num = len(device)
            self.train_net = DistributedNet(network, config,
                                            is_training=True,
                                            device=device, name=name, reuse=None)
            self.eval_net = DistributedNet(network, config,
                                           is_training=False,
                                           device=device, name=name, reuse=True)
        else:
            self.distributed_num = 1
            self.train_net = network(config, is_training=True, device=device, name=name, reuse=None)
            self.eval_net = network(config, is_training=False, device=device, name=name, reuse=True)

        # q distribution in TRF
        if config.reference_model is None:
            self.q_model = None
        else:
            model_path = config.reference_model.split(':')[-1]
            print('[TRF] load reference : %s' % model_path)
            self.q_model = lstmlm.LM.load(model_path, device=device)

        # create noise model
        if self.config.noise_sampler.find('gram') != -1:
            ngram = int(self.config.noise_sampler.split('gram')[0])
            if ngram == 1:
                self.noise_sampler = NoiseSamplerUnigram(config, data)
            else:
                if self.config.noise_sampler.find('train') != -1:
                    self.noise_sampler = NoiseSamplerNgramTrain(config, data, ngram)
                else:
                    self.noise_sampler = NoiseSamplerNgram(config, data, ngram)
        elif self.config.noise_sampler.find('lstm:') != -1:
            lstm_path = self.config.noise_sampler.split(':')[-1]
            self.noise_sampler = NoiseSamplerLSTMEval(config, data, lstm_path, device=device)

        # summary
        self.summ_vars = layers.SummaryVariables()
        self.grad_norm = tf.global_norm(self.train_net.grads_list[0])

        # saver
        self.saver = tf.train.Saver()
        self.is_load_model = False

        # debuger
        self.write_files = wb.FileBank()

        # time recorder
        self.time_recoder = wb.clock()

    def parser_net(self, net_type):
        if net_type == 'cnn':
            network = NetCNN
        elif net_type == 'rnn':
            network = NetRnn
        elif net_type == 'mix':
            network = NetMix
        else:
            raise TypeError('undefined structure type= ' + net_type)
        return network

    def set_session(self, session):
        self.session = session

    def get_session(self):
        if self.session is None:
            self.session = tf.get_default_session()
            if self.session is None:
                raise TypeError('session is None, please call set_session() to set session')
        return self.session

    def initialize(self, session):
        self.set_session(session)
        if self.q_model is not None:
            self.q_model.restore(session)

    def phi(self, inputs, lengths):
        return self.eval_net.run_phi(self.get_session(), inputs, lengths) + self.get_qlogps(inputs, lengths)

    def logps(self, inputs, lengths):
        return self.eval_net.run_logps(self.get_session(),
                                       inputs, lengths,
                                       self.get_qlogps(inputs, lengths))

    def get_qlogps(self, inputs, lengths):
        if self.q_model is not None:
            q_logps = self.q_model.conditional(self.get_session(), inputs, 1, lengths, initial_state=True)
        else:
            q_logps = np.zeros(len(lengths))
        return q_logps

    def get_log_probs(self, seq_list, is_norm=True):
        batch_size = (1 + self.config.noise_factor) * self.config.batch_size
        logprobs = np.zeros(len(seq_list))

        if is_norm:
            for i in range(0, len(seq_list), batch_size):
                logprobs[i: i+batch_size] = self.logps(
                    *reader.produce_data_to_trf(seq_list[i: i+batch_size])
                )
        else:
            for i in range(0, len(seq_list), batch_size):
                logprobs[i: i+batch_size] = self.phi(
                    *reader.produce_data_to_trf(seq_list[i: i+batch_size])
                )
        return logprobs

    def true_logz(self, max_len=None):
        if max_len is None:
            max_len = self.config.max_len

        logz = np.zeros(max_len - self.config.min_len + 1)
        for l in range(self.config.min_len, max_len+1):
            x_batch = [x for x in sp.SeqIter(l, self.config.vocab_size,
                                             beg_token=self.config.beg_token,
                                             end_token=self.config.end_token)]
            logz[l-self.config.min_len] = sp.log_sum(self.get_log_probs(x_batch, False))
        return logz

    def eval(self, data_list):
        logps = self.get_log_probs(data_list)

        lens = [len(x) - int(self.config.beg_token is not None) for x in data_list]
        s = - sum(logps)
        nll = s / len(data_list)
        ppl = np.exp(s / sum(lens))
        return nll, ppl

    def noise_generate_all(self, data_seqs, train_batch):

        with self.time_recoder.recode('noise_sample'):
            noise_seqs, noise_logps = self.noise_sampler.get()

        with self.time_recoder.recode('data_logp'):
            data_logps = self.noise_sampler.noise_logps(data_seqs)

        assert len(noise_seqs) / len(data_seqs) == self.config.noise_factor

        all_seqs = data_seqs + noise_seqs
        with self.time_recoder.recode('q_logps'):
            inputs, lengths = reader.produce_data_to_trf(all_seqs)
            q_logps = self.get_qlogps(inputs, lengths)

        return all_seqs, np.concatenate([data_logps, noise_logps]), q_logps

    def update(self, data_seqs, train_size):

        with self.time_recoder.recode('noise_sample_logp'):
            total_seqs, noise_logps, q_logps = self.noise_generate_all(data_seqs, train_size)
            inputs, lengths = reader.produce_data_to_trf(total_seqs)

        # compute the debug before update parameters
        if self.config.write_dbg:
            with self.time_recoder.recode('cmp_dbg'):
                # run values
                op_dict = {'dbg': self.train_net.dbg,
                           # 'vars': self.train_net.vars,
                           # 'grad': self.train_net.grad_list[0]
                           'data_logps': self.train_net.logps,
                           'data_phis': self.train_net.phi,
                           'cluster_prob': self.train_net.dbg['P(C)']
                           }
                dbg_dict = self.train_net.run(self.get_session(), inputs, lengths,
                                              q_logps, noise_logps,
                                              ops=op_dict)

            with self.time_recoder.recode('write_dbg'):
                # write noise sequence
                f = self.write_files.get('noise', os.path.join(self.logdir, self.name) + '.noise')
                for i in range(len(total_seqs)):
                    if i < len(data_seqs):
                        f.write('[data]  ')
                    else:
                        f.write('[noise] ')
                    f.write('p={:.4f}  logpm={:.2f}  logpn={:.2f}  phi={:.2f} \t'.format(
                        dbg_dict['cluster_prob'][i],
                        dbg_dict['data_logps'][i], noise_logps[i],
                        dbg_dict['data_phis'][i],
                    ))
                    f.write(' '.join([str(w) for w in total_seqs[i]]) + '\n')
                f.flush()

                # dbg probs
                f = self.write_files.get('prob', os.path.join(self.logdir, self.name) + '.prob')
                f.write('step=%d\n' % self.get_session().run(self.train_net.global_step))
                np.set_printoptions(threshold=2000)
                for key, v in dbg_dict['dbg'].items():
                    s = np.array2string(v, formatter={'float_kind': lambda x: "%.2f" % x})
                    f.write(key + '=' + s + '\n')
                f.flush()

                # self.write_vars.write('step=%d\n' % self.get_session().run(self.train_net.global_step))
                # for v, var in zip(self.train_net.vars, res_dict['vars']):
                #     self.write_vars.write(v.name + '\n')
                #     self.write_vars.write(str(var) + '\n')
                # self.write_vars.flush()
                #
                # self.write_grad.write('step=%d\n' % self.get_session().run(self.train_net.global_step))
                # for v, var in zip(self.train_net.vars, res_dict['grad']):
                #     self.write_grad.write(v.name + '\n')
                #     self.write_grad.write(str(var) + '\n')
                # self.write_grad.flush()

        #################################
        # update paraemters
        #################################
        with self.time_recoder.recode('update'):
            # update parameters
            if self.config.update_zeta:
                op_dict = {'train_zeta': self.train_net.train_zeta}
            else:
                op_dict = {}

            op_dict['train_params'] = self.train_net.train_op
            op_dict['loss'] = self.train_net.loss
            op_dict['grad_logz'] = self.train_net.grad_for_zeta[0]
            op_dict['grad_norm'] = self.grad_norm

            res_dict = self.train_net.run(self.get_session(), inputs, lengths,
                                          q_logps, noise_logps,
                                          op_dict)
            # update logz1
            # logz1 = self.true_logz(self.config.min_len)[0]
            # self.train_net.set_logz_base(self.get_session(), logz1)

        # write zeta
        with self.time_recoder.recode('write_zeta'):
            logz = self.train_net.get_logz(self.get_session())
            f = self.write_files.get('zeta', os.path.join(self.logdir, self.name) + '.zeta')
            f.write('logz=' + ' '.join(['{:.3f}'.format(i) for i in logz]) + '\n')
            f.write('grad=' + ' '.join(['{:.3f}'.format(i) for i in res_dict['grad_logz']]) + '\n')
            spi = np.zeros(self.config.max_len-self.config.min_len + 1)
            for i in lengths:
                spi[i - self.config.min_len] += 1
            spi /= np.sum(spi)
            f.write('pi=  ' + ' '.join(['{:.3f}'.format(i) for i in spi]) + '\n')
            f.write('\n')
            f.flush()

        return res_dict

    def save(self, logname):
        """save mode to dirs"""
        print('[TRF] save ckpt to %s' % logname)
        self.saver.save(self.get_session(), logname + '.ckpt')

    def load(self, logname):
        """save mode to dirs"""
        if wb.exists(logname + '.ckpt.index'):
            print('[TRF] load ckpt from %s' % logname)
            self.saver.restore(self.get_session(), logname + '.ckpt')
            self.is_load_model = True

    def train(self, sv, session, nbest=None, lmscale_vec=None,
              print_per_epoch=0.1,
              wer_per_epoch=1.0,
              lmscore_per_epoch=1,
              model_per_epoch=50,
              load_model_epoch=None,
              operation=None):

        # start noise
        self.noise_sampler.start()

        self.set_session(session)
        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]
        test_list = self.data.datas[2]
        print('[TRF] train_list={:,}'.format(len(train_list)), 'valid_list={:,}'.format(len(valid_list)))
        print('[TRF] param_size={:,}'.format(session.run(self.train_net.var_size)))

        # create folder to store the lmscore ans models
        logname = os.path.join(self.logdir, self.name)
        wb.mkdir(os.path.join(self.logdir, 'lmscore'))
        wb.mkdir(os.path.join(self.logdir, 'models'))
        logname_lmscore = os.path.join(self.logdir, 'lmscore/' + self.name)
        logname_models = os.path.join(self.logdir, 'models/' + self.name)

        # load models
        if load_model_epoch is None:
            # load the last model
            self.load(logname)
        else:
            self.load(logname_models + '.epoch{}'.format(load_model_epoch))

        print('[TRF] [Train]...')
        epoch_contain_step = len(train_list) // self.config.batch_size

        time_beginning = time.time()
        model_train_nll = []
        model_train_loss = []

        step = session.run(self.train_net.global_step)
        epoch = step * self.config.batch_size / len(train_list)
        print_next_epoch = int(epoch)
        wer_next_epoch = int(epoch)
        while epoch < self.config.max_epoch:

            if step % epoch_contain_step == 0:
                np.random.shuffle(train_list)

            # learining rate
            lr_param = self.config.lr_param.get_lr(step + 1, epoch)
            lr_zeta = self.config.lr_zeta.get_lr(step + 1, epoch)
            self.train_net.run_set_lr(session, lr_param, lr_zeta)

            # current data sequences
            data_seqs = train_list[
                        step % epoch_contain_step * self.config.batch_size:
                        (step % epoch_contain_step + 1) * self.config.batch_size
                        ]

            # update parameters
            res_dict = self.update(data_seqs, len(train_list))

            # compute the nll on training set
            with self.time_recoder.recode('train_eval'):
                model_train_nll.append(self.eval(data_seqs)[0])
                model_train_loss.append(res_dict['loss'])

            # update steps
            step = session.run(self.train_net.global_step)
            epoch = step * self.config.batch_size / len(train_list)

            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]
                    model_test_nll = self.eval(test_list)[0]

                time_since_beg = (time.time() - time_beginning) / 60

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                info['lr_param'] = '{:.3e}'.format(lr_param)
                info['lr_zeta'] = '{:.3e}'.format(lr_zeta)
                info['norm'] = res_dict['grad_norm']
                info['logz1'] = self.true_logz(self.config.min_len)[0]
                info['loss'] = np.mean(model_train_loss[-epoch_contain_step:])
                info['train'] = np.mean(model_train_nll[-epoch_contain_step:])
                info['valid'] = model_valid_nll
                info['test'] = model_test_nll
                trfbase.print_line(info)

                with self.time_recoder.recode('summary'):
                    self.summ_vars.write_summary(sv, session)

                print('[end]')

                ######################################
                # calculate the WER
                #####################################
                if epoch >= wer_next_epoch and nbest is not None:
                    wer_next_epoch = int(epoch + wer_per_epoch)

                    # resocring
                    with self.time_recoder.recode('rescore'):
                        time_beg = time.time()
                        nbest.lmscore = -self.get_log_probs(nbest.get_nbest_list(self.data))
                        rescore_time = time.time() - time_beg

                    # compute wer
                    with self.time_recoder.recode('wer'):
                        time_beg = time.time()
                        if lmscale_vec is not None:
                            wer = nbest.wer(lmscale_vec)
                        else:
                            wer = nbest.wer()
                        wer_time = time.time() - time_beg

                    print('epoch={:.2f} test_wer={:.2f} lmscale={} '
                          'rescore_time={:.2f}, wer_time={:.2f}'.format(
                           epoch, wer, nbest.lmscale,
                           rescore_time / 60, wer_time / 60))

                    res = wb.FRes(os.path.join(self.logdir, 'wer_per_epoch.log'))
                    res_name = 'epoch%.2f' % epoch
                    res.Add(res_name, ['lm-scale'], [nbest.lmscale])
                    res.Add(res_name, ['wer'], [wer])

                    # write models
                    self.save(logname_models + '.epoch%.2f' % epoch)

                    # write lmscore
                    nbest.write_lmscore(logname_lmscore + '.epoch%.2f.lmscore' % epoch)

                #####################################
                # write time
                #####################################
                # write to file
                f = self.write_files.get('time', os.path.join(self.logdir, self.name) + '.time')
                f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, time_since_beg))
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.time_recoder.items()]) + '\n')
                f.flush()

            ###########################
            # extra operations
            ###########################
            if operation is not None:
                operation.run(step, epoch)

            # write feat after each epoch
            if step % epoch_contain_step == 0:
                self.save(logname)
                if model_per_epoch is not None and int(epoch) % model_per_epoch == 0:
                    self.save(logname_models + '.epoch{}'.format(int(epoch)))

        # stop the sub-process
        self.noise_sampler.release()

