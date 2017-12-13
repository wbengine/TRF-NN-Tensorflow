import tensorflow as tf
from copy import deepcopy
import numpy as np
import time
import json
import os

from base import *
from trf import trfbase
from trf import trfjsa


class Config(trfjsa.Config):
    def __init__(self, data):
        super().__init__(data)
        # for structure
        self.embedding_dim = 128
        self.load_embedding_path = None
        self.cnn_filters = [(i, 128) for i in range(1, 6)]
        self.cnn_layers = 3
        self.cnn_hidden = 128
        self.cnn_width = 3
        self.cnn_activation = 'relu'
        self.cnn_skip_connection = True
        self.cnn_residual = False
        self.dropout = 0
        self.init_weight = 0.1
        self.batch_normalize = False

        self.block_layers = 5
        self.max_grad_norm = None

        self.update_zeta = True
        self.update_param = True

    # def initial_zeta(self):
    #     return np.zeros(self.max_len+1)

    def __str__(self):
        s = 'trf_cnn'
        if self.batch_normalize:
            s += '_BN'
        s += '_e{}'.format(self.embedding_dim)
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

        return s


class Net(object):
    def __init__(self, config, is_training, is_propose=False, device='/gpu:0', name='phi_net', reuse=None):
        """

        Args:
            config: the config of TRF
            is_training: if True, then compute the gradient of potential function with respect to parameters
            is_propose: if True, then compute a propose model used to perform MCMC
            device: the device
            name: the name
            reuse: if resuse the variables
        """
        self.is_training = is_training
        self.config = config
        input_batch_size = self.config.train_batch_size + self.config.sample_batch_size if is_training else None

        initializer = tf.random_uniform_initializer(-config.init_weight, config.init_weight)
        with tf.device(device), tf.variable_scope(name, reuse=reuse, initializer=initializer):
            # inputs: of shape (batch_size, seq_len)
            # lengths: of shape (batch_size,)
            # extra_weight: of shape (batch_size,), used to input the weight of such as ngram features
            self._inputs = tf.placeholder(tf.int32, [input_batch_size, None], name='inputs')
            self._lengths = tf.placeholder(tf.int32, [input_batch_size], name='lengths')
            self._extra_weight = tf.placeholder(tf.float32, [input_batch_size], name='extra_weight')
            #############################################################
            # construct the network
            #############################################################
            outputs, self.vars = self.output(config, self._inputs, self._lengths, reuse=reuse)
            # phi
            self.phi = outputs + self._extra_weight

            # print variables
            if reuse is None:
                print('variables in %s' % name)
                for v in self.vars:
                    print('\t' + v.name, v.shape, v.device)

            #############################################################
            # define zeta and compute the log prob
            #############################################################
            valid_len = config.max_len - config.min_len + 1
            self.pi = tf.get_variable('pi', shape=[valid_len], dtype=tf.float32,
                                      trainable=False,
                                      initializer=tf.constant_initializer(config.pi_0[config.min_len:]))
            self.zeta = tf.get_variable('zeta', shape=[valid_len], dtype=tf.float32,
                                        trainable=False,
                                        initializer=tf.constant_initializer(config.init_zeta))
            self.logz_base = tf.get_variable('logz_base', shape=[], dtype=tf.float32, trainable=False,
                                             initializer=tf.constant_initializer(np.log(self.config.vocab_size)))
            # compute the log prob
            norm_constant = self.zeta + self.logz_base
            self.logps = self.phi + tf.log(tf.gather(self.pi, self._lengths - config.min_len)) - \
                         tf.gather(norm_constant, self._lengths - config.min_len)

            # set pi
            self._new_pi = tf.placeholder(tf.float32, [valid_len])
            self._set_pi = tf.assign(self.pi, self._new_pi)
            self._new_zeta = tf.placeholder(tf.float32, [valid_len])
            self._set_zeta = tf.assign(self.zeta, self._new_zeta)

            #############################################################
            # Gradient for parameters
            #############################################################
            # inputs[0: _train_num] is the training sequences
            # inputs[_train_num:  ] is the sampling sequences
            self._train_num = tf.placeholder(tf.int32, shape=[], name='train_num')
            self.grads_to_params = self.compute_gradient_to_params(outputs[0: self._train_num],
                                                                   outputs[self._train_num:],
                                                                   self._lengths[self._train_num:])
            self.grads_to_zeta = self.compute_gradient_to_zeta(self._lengths[self._train_num:])

            #############################################################
            # update parameters
            #############################################################
            if is_training:
                self._lr = [tf.Variable(1.0, trainable=False, name='learning_rate_param'),
                            tf.Variable(1.0, trainable=False, name='learning_rate_zeta')]
                self.global_step = tf.Variable(0, trainable=False, name='global_step')

                # update parameters
                if config.opt_method.lower() == 'adam':
                    optimizer = tf.train.AdamOptimizer(self._lr[0])
                elif config.opt_method.lower() == 'adagrad':
                    optimizer = tf.train.AdagradOptimizer(self._lr[0])
                elif config.opt_method.lower() == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(self._lr[0])
                else:
                    raise TypeError('undefined method ' + config.opt_method)

                if config.max_grad_norm is not None:
                    self.grads_to_params, _ = tf.clip_by_global_norm(self.grads_to_params, config.max_grad_norm)

                batch_norm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                print('BN_update_ops in:', name)
                for op in batch_norm_updates:
                    print('\t' + op.name, op.shape, op.device)
                with tf.control_dependencies(batch_norm_updates):
                    self.update_params = optimizer.apply_gradients(zip(self.grads_to_params, self.vars),
                                                                   global_step=self.global_step)

                # update zeta
                zeta_step = tf.minimum(self.grads_to_zeta * self._lr[1], config.zeta_gap)
                self.update_zeta = tf.assign_add(self.zeta, zeta_step)
                self.update_zeta = tf.assign_sub(self.update_zeta, self.update_zeta[0] * tf.ones_like(self.update_zeta))

                # update logz_base
                self._new_logz_base = tf.placeholder(tf.float32, [])
                self._set_logz_base = tf.assign(self.logz_base, self._new_logz_base)

                # set learning rate
                self._new_lr = tf.placeholder(tf.float32, [])
                self._set_lr = [tf.assign(self._lr[i], self._new_lr) for i in range(len(self._lr))]

    def activation(self, x, reuse, name='BN', add_activation=True):
        if self.config.batch_normalize:
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
        pre_net_w = tf.get_variable('pre_net_w',
                                    [embedding_dim, config.cnn_hidden], dtype=tf.float32)
        pre_net_b = tf.get_variable('pre_net_b',
                                    [config.cnn_hidden], dtype=tf.float32)
        emb_output = tf.nn.relu(
            tf.matmul(tf.reshape(emb_output, [-1, embedding_dim]), pre_net_w) + pre_net_b)
        emb_output = tf.reshape(emb_output, [batch_size, max_len, config.cnn_hidden])
        emb_output = emb_output * expand_len_mask

        inputs = emb_output
        tf.summary.image('embedding', tf.expand_dims(inputs, axis=-1), max_outputs=4, collections=['cnn'])
        tf.summary.histogram('embedding', word_embedding, collections=['cnn'])

        # dropout
        if self.is_training and self.config.dropout > 0:
            inputs = tf.nn.dropout(inputs, keep_prob=1. - self.config.dropout)

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

            conv = self.activation(conv, reuse=reuse, name='cnn0_{}_BN'.format(filter_width))

            conv_list.append(conv * expand_len_mask)

        if conv_list:
            inputs = tf.concat(conv_list, axis=-1)
        tf.summary.image('cnn0', tf.expand_dims(inputs, axis=-1), max_outputs=4, collections=['cnn'])

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
            conv_output = self.activation(conv_output, reuse=reuse, name='cnn{}_BN'.format(i+1)) * expand_len_mask
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
            inputs = self.activation(tf.add_n(skip_connections), reuse=reuse, name='skip_conn')

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

    def compute_gradient_to_params(self, train_phi, sample_phi, sample_lengths):
        train_loss = tf.reduce_sum(train_phi) / self.config.train_batch_size
        # train_grads = tf.gradients(train_loss, self.vars)

        pi_rate = self.config.pi_true[self.config.min_len:] / self.config.pi_0[self.config.min_len:]
        weights = tf.gather(pi_rate.astype('float32'), sample_lengths - self.config.min_len)
        sample_loss = tf.reduce_sum(sample_phi * weights) / self.config.sample_batch_size
        # sample_grads = tf.gradients(sample_loss, self.vars)

        grads = tf.gradients(sample_loss - train_loss, self.vars)

        return grads

    def compute_gradient_to_zeta(self, sample_lengths):
        len_sum = tf.scatter_nd(tf.reshape(sample_lengths - self.config.min_len, [-1, 1]),
                                tf.ones(shape=tf.shape(sample_lengths), dtype=tf.float32),
                                shape=[self.config.max_len - self.config.min_len + 1])
        zeta_grads = len_sum / self.config.sample_batch_size / self.config.pi_0[self.config.min_len:]
        return zeta_grads

    def get_logz(self, session):
        logz_base = session.run(self.logz_base)
        zeta = np.append(np.zeros(self.config.min_len),
                         session.run(self.zeta) + logz_base)
        return zeta

    def get_zeta(self, session):
        return np.append(np.zeros(self.config.min_len), session.run(self.zeta))

    def set_zeta(self, session, zeta):
        session.run(self._set_zeta, {self._new_zeta: zeta[self.config.min_len:]})

    def set_pi(self, session, pi):
        session.run(self._set_pi, {self._new_pi: pi[self.config.min_len:]})

    def set_logz_base(self, session, logz_base):
        return session.run(self._set_logz_base, {self._new_logz_base: logz_base})

    def run_phi(self, session, inputs, lengths, extra_weight=None):
        if extra_weight is None:
            extra_weight = np.zeros(len(inputs), dtype='float32')
        return session.run(self.phi,
                           {self._inputs: inputs,
                            self._lengths: lengths,
                            self._extra_weight: extra_weight})

    def run_logps(self, session, inputs, lengths, extra_weight=None):
        if extra_weight is None:
            extra_weight = np.zeros(len(inputs), dtype='float32')
        return session.run(self.logps,
                           {self._inputs: inputs,
                            self._lengths: lengths,
                            self._extra_weight: extra_weight})

    def run(self, session, inputs, lengths, train_num, ops):
        return session.run(ops,
                           {self._inputs: inputs,
                            self._lengths: lengths,
                            self._train_num: train_num})

    def run_train(self, session, inputs, lengths, train_num):
        ops = []
        if self.config.update_param:
            ops.append(self.update_params)
        if self.config.update_zeta:
            ops.append(self.update_zeta)
        self.run(session, inputs, lengths, train_num, ops=ops)

    def run_set_lr(self, session, lr1, lr2):
        session.run(self._set_lr[0], {self._new_lr: lr1})
        session.run(self._set_lr[1], {self._new_lr: lr2})


class TRF(trfjsa.FastTRF):
    def __init__(self, config, data, name='trf', logdir='trf', device='/gpu:0',
                 simulater_device=None, network=Net,
                 q_model=None):
        super().__init__(config, data, name=name, logdir=logdir,
                         simulater_device=device if simulater_device is None else simulater_device)

        with tf.name_scope('Train'):
            self.train_net = network(config, True, device=device, name=name, reuse=None)
        with tf.name_scope('Eval'):
            self.eval_net = network(config, False, device=device, name=name, reuse=True)

        self.q_model = q_model

        # summarys:
        self.param_size = tf.add_n([tf.size(v) for v in self.train_net.vars])
        self.param_norm = tf.global_norm(self.train_net.vars)
        self._global_step = self.train_net.global_step

        # summarys
        var_summ = layers.SummaryVariables()
        self._pure_summary = tf.summary.merge([
                                               tf.summary.scalar('trf/param_norm', self.param_norm)]
                                              )

        # saver
        self.saver = tf.train.Saver()
        self.is_load_model = False

    def get_qlogps(self, inputs, lengths):
        if self.q_model is not None:
            q_logps = self.q_model.conditional(self.get_session(), inputs, 1, lengths, initial_state=True)
        else:
            q_logps = np.zeros(len(lengths))
        return q_logps

    def phi(self, inputs, lengths):
        extra_weight = super().phi(inputs, lengths)  # weight of discrete features
        extra_weight += self.get_qlogps(inputs, lengths)
        return self.eval_net.run_phi(self.get_session(), inputs, lengths, extra_weight)

    def logps(self, inputs, lengths):
        extra_weight = super().phi(inputs, lengths)  # weight of discrete features
        extra_weight += self.get_qlogps(inputs, lengths)
        return self.eval_net.run_logps(self.get_session(), inputs, lengths, extra_weight)

    def get_log_probs(self, seq_list, is_norm=True):
        batch_size = self.config.batch_size

        splited_seq_list, splited_index = self.data.cut_data_to_length(seq_list, self.config.max_len)
        logprobs = np.zeros(len(splited_seq_list))

        if is_norm:
            self.eval_net.set_pi(self.get_session(), self.config.pi_true)

            for i in range(0, len(splited_seq_list), batch_size):
                logprobs[i: i + batch_size] = self.logps(
                    *reader.produce_data_to_trf(splited_seq_list[i: i + batch_size])
                )

            self.eval_net.set_pi(self.get_session(), self.config.pi_0)
        else:
            assert len(seq_list) == len(splited_seq_list)
            for i in range(0, len(splited_seq_list), batch_size):
                logprobs[i: i + batch_size] = self.phi(
                    *reader.produce_data_to_trf(splited_seq_list[i: i + batch_size])
                )

        # merge the logprobs
        res_logps = np.array([np.sum(logprobs[i: j]) for i, j in splited_index])

        return res_logps

    def true_normalize_all(self):
        super().true_normalize_all()
        self.train_net.set_logz_base(self.get_session(), self.logz[self.config.min_len])
        self.train_net.set_zeta(self.get_session(), self.zeta)

    def update_zeta(self, sample_list, lr_zeta):
        # update zeta
        sample_pi = np.zeros(self.config.max_len + 1)
        for seq in sample_list:
            sample_pi[len(seq)] += 1.
        sample_pi /= len(sample_list)

        self.sample_acc_count += sample_pi * len(sample_list)
        self.sample_cur_pi = sample_pi

        self.logz = self.train_net.get_logz(self.get_session())
        self.zeta = self.train_net.get_zeta(self.get_session())

    def update(self, session, train_seqs, sample_seqs, global_step, global_epoch):
        # assert len(train_seqs) % self.config.batch_size == 0
        # assert len(sample_seqs) % self.config.batch_size == 0

        lr_param = self.config.lr_param.get_lr(global_step, global_epoch)
        lr_zeta = self.config.lr_zeta.get_lr(global_step, global_epoch)
        lr_cnn = self.config.lr_cnn.get_lr(global_step, global_epoch)

        # set lr
        self.train_net.run_set_lr(self.get_session(), lr_cnn, lr_zeta)

        # update parameters
        inputs, lengths = reader.produce_data_to_trf(train_seqs + sample_seqs)
        self.train_net.run_train(self.get_session(), inputs, lengths, len(train_seqs))

        # update logz_base
        self.train_net.set_logz_base(self.get_session(), self.true_normalize(self.config.min_len))

        # update super
        super().update(session, train_seqs, sample_seqs, global_step, global_epoch)

        return {'lr_cnn': lr_cnn, 'lr_param': lr_param, 'lr_zeta': lr_zeta}

    def eval(self, data_list):
        self.eval_net.set_pi(self.get_session(), self.config.pi_true)
        logps = self.get_log_probs(data_list)
        self.eval_net.set_pi(self.get_session(), self.config.pi_0)

        lens = [len(x) - int(self.config.beg_token is not None) for x in data_list]
        s = - sum(logps)
        nll = s / len(data_list)
        ppl = np.exp(s / sum(lens))
        return nll, ppl

    def eval_pi0(self, data_list):
        logps = self.get_log_probs(data_list)

        lens = [len(x) - int(self.config.beg_token is not None) for x in data_list]
        s = - sum(logps)
        nll = s / len(data_list)
        ppl = np.exp(s / sum(lens))
        return nll, ppl

    def save(self, logname):
        """save mode to dirs"""
        super().save(logname)
        print('[TRF] save ckpt to %s' % logname)
        self.saver.save(self.get_session(), logname + '.ckpt')

    def load(self, logname):
        """save mode to dirs"""
        super().load(logname)
        if wb.exists(logname + '.ckpt.index'):
            print('[TRF] load ckpt from %s' % logname)
            self.saver.restore(self.get_session(), logname + '.ckpt')
            self.is_load_model = True

    def prepare(self):
        super().prepare()
        print('[TRF] nn.param_num={:,}'.format(self.get_session().run(self.param_size)))

    def train_after_update(self, **argv):
        pass

    # def draw(self, n):
    #     pass
