import tensorflow as tf
from copy import deepcopy
import numpy as np
import time
import json
import os

from . import layers
from . import reader
from . import wblib as wb
from . import trfbase
from . import word2vec


class Config(trfbase.Config):
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
        self.cnn_final = 'linear'  # one of 'cnn', 'fc', 'vc'
        self.cnn_shared_over_layers = False
        self.cnn_skip_connection = False
        self.cnn_residual = False
        self.pretrain_net_path = None


class Net(object):
    def __init__(self, config, is_training, is_empirical, is_pretrain=False,
                 device='/gpu:0', name='phi_net', reuse=None):
        """

        Args:
            config: the config of TRF
            is_training: if True, then compute the gradient of potential function with respect to parameters
            is_empirical: if Ture, compute the empirical variance
            is_pretrain: if True, compute the softmax to compute the loss, used to pretrain
            device: the device
            name: the name
            reuse: if resuse the variables
        """
        self.is_training = is_training
        self.is_empirical = is_empirical
        self.is_pretrain = is_pretrain
        self.config = config

        initializer = tf.random_uniform_initializer(-config.init_weight, config.init_weight)
        with tf.device(device), tf.variable_scope(name, reuse=reuse, initializer=initializer):
            # inputs: of shape (batch_size, seq_len)
            # lengths: of shape (batch_size,)
            # extra_weight: of shape (batch_size,), used to input the weight of such as ngram features
            self._inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
            self._lengths = tf.placeholder(tf.int32, [None], name='lengths')

            #############################################################
            # construct the network
            #############################################################
            self.loss = tf.no_op()
            outputs, self.vars = self.output(config, reuse=reuse)
            self._extra_weight = tf.placeholder(tf.float32, [None], name='extra_weight')
            self._phi = outputs + self._extra_weight

            ##############################################################
            # compute the expectation
            # scalars: of shape [batch_sizs]
            ##############################################################
            # if training, create varialbes to store the gradients for given batches
            if is_training:
                self._scalars = tf.placeholder(tf.float32, [None], name='scalars')
                self.expectation, self.clean_expec, self._update_expec = \
                    self.compute_expectation(outputs, self._scalars)

            if is_training and is_empirical:
                self.expectation2, self.clean_expec2, self._update_expec2 = \
                    self.compute_expectation2(outputs, self._scalars)

            ##############################################################
            # to pretrain the parameters
            ##############################################################
            if is_pretrain:
                self.train_layer = layers.TrainOp(self.loss, self.vars,
                                                  optimize_method=config.opt_method,
                                                  max_grad_norm=config.max_grad_norm)
                self._train_op = self.train_layer.train_op
            # used to write or restore the parameters
            self.pretrain_saver = tf.train.Saver(self.vars)

            # pi and zeta
            valid_len = config.max_len - config.min_len + 1
            self._pi_0 = tf.get_variable('pi', shape=[valid_len], dtype=tf.float32,
                                         trainable=False,
                                         initializer=tf.constant_initializer(config.pi_0[config.min_len:]))
            self._zeta = tf.get_variable('zeta', shape=[valid_len], dtype=tf.float32,
                                         trainable=False,
                                         initializer=tf.constant_initializer(config.initial_zeta()[config.min_len:]))
            self._logz_base = tf.get_variable('logz_base', shape=[], dtype=tf.float32, trainable=False)
            norm_constant = tf.log(self._pi_0) - self._zeta - self._logz_base
            self._logp = self._phi + tf.gather(norm_constant, self._lengths - config.min_len)
            # setting
            self._new_pi_or_zeta = tf.placeholder(tf.float32, shape=[config.max_len+1], name='new_pi_or_zeta')
            self._set_pi_0 = tf.assign(self._pi_0, self._new_pi_or_zeta[config.min_len:])
            self._set_zeta = tf.assign(self._zeta, self._new_pi_or_zeta[config.min_len:])
            self._new_float = tf.placeholder(tf.float32, shape=[], name='new_float')
            self._set_logz_base = tf.assign(self._logz_base, self._new_float)

            # update zeta
            self._sample_pi = tf.placeholder(tf.float32, shape=[config.max_len+1], name='sample_pi')
            self._zeta_lr = tf.placeholder(tf.float32, shape=[], name='zeta_lr')
            zeta_step = tf.minimum(self._zeta_lr *
                                   self._sample_pi[config.min_len:] / self._pi_0,
                                   config.zeta_gap)
            self._update_zeta = tf.assign_add(self._zeta, zeta_step)
            self._update_zeta = tf.assign_sub(self._update_zeta,
                                              self._update_zeta[0] * tf.ones_like(self._update_zeta))

            # summary
            tf.summary.scalar('logz_base', self._logz_base, collections=['cnn'])
            self.summary_image = tf.summary.merge_all('cnn')

    def output(self, config, reuse=None):
        """
        Using the self._inputs and self._lengths to calculate phi of TRF
        """
        vocab_size = config.vocab_size
        embedding_dim = config.embedding_dim

        if config.cnn_activation is None:
            cnn_activation = None
        elif config.cnn_activation == 'relu':
            cnn_activation = tf.nn.relu
        elif config.cnn_activation == 'tanh':
            cnn_activation = tf.nn.tanh
        else:
            raise TypeError('unknown activation {}'.format(config.cnn_activation))

        batch_size = tf.shape(self._inputs)[0]
        max_len = tf.shape(self._inputs)[1]
        # the length mask, the position < len is 1., otherwise is 0.
        len_mask = tf.tile(tf.reshape(tf.range(max_len, dtype=tf.int32), [1, max_len]), [batch_size, 1])
        len_mask = tf.less(len_mask, tf.reshape(self._lengths, [batch_size, 1]))
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
        emb_output = tf.nn.embedding_lookup(word_embedding, self._inputs)  # (batch_size, seq_len, emb_dim)

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
                activation=cnn_activation,
                reuse=reuse,
                name='cnn0_{}'.format(filter_width)
            )
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

            if config.cnn_shared_over_layers:
                conv_output = tf.layers.conv1d(
                    inputs=inputs,
                    filters=config.cnn_hidden,
                    kernel_size=config.cnn_width,
                    padding='same',
                    activation=cnn_activation,
                    reuse=(None if reuse is None and i == 0 else True),
                    name='cnn'
                )
            else:
                conv_output = tf.layers.conv1d(
                    inputs=inputs,
                    filters=config.cnn_hidden,
                    kernel_size=config.cnn_width,
                    padding='same',
                    activation=cnn_activation,
                    reuse=reuse,
                    name='cnn{}'.format(i + 1)
                )
            conv_output = conv_output * expand_len_mask
            tf.summary.image('cnn{}'.format(i+1),
                             tf.expand_dims(conv_output, axis=-1),
                             max_outputs=4, collections=['cnn'])

            if config.cnn_skip_connection:
                skip_scalar = tf.get_variable(name='cnn{}_skip_scalar'.format(i + 1),
                                              shape=[config.cnn_hidden], dtype=tf.float32)
                skip_connections.append(conv_output * tf.reshape(skip_scalar, [1, 1, config.cnn_hidden]))

            inputs = conv_output

        # skip connections
        if skip_connections:
            inputs = tf.nn.relu(tf.add_n(skip_connections))

        # residual connection
        if config.cnn_residual:
            inputs = tf.nn.relu(emb_output + inputs)

        tf.summary.image('cnn_end',
                         tf.expand_dims(inputs, axis=-1),
                         max_outputs=4, collections=['cnn'])

        ### TRF outputs
        # final conv
        if config.cnn_final == 'linear':
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
        elif config.cnn_final == 'vc':
            output_weight = tf.get_variable('output_weight', [config.cnn_hidden, vocab_size], dtype=tf.float32)
            logits = tf.matmul(tf.reshape(inputs, [-1, config.cnn_hidden]),
                               output_weight)  # of shape (batch_size*seq_len, vocab_size)
            logits = tf.reshape(logits, [batch_size, -1, vocab_size])
            one_hot = tf.one_hot(self._inputs, depth=vocab_size)
            outputs = tf.reduce_sum(logits * one_hot, axis=-1)
        elif config.cnn_final == 'fc':
            fc_weight = tf.get_variable('fc_weight', [config.max_len, config.cnn_hidden], dtype=tf.float32)
            outputs = tf.reduce_sum(inputs * fc_weight[0: max_len], axis=-1)
        else:
            raise TypeError('config.cnn_final should be one of \'linear\', \'fc\', \'vc\'')

        outputs = outputs * len_mask
        outputs = tf.reduce_sum(outputs, axis=-1)  # of shape [batch_size]

        if self.is_pretrain:
            # pre-train loss function
            softmax = layers.Softmax(inputs, self._inputs, vocab_size, 1)
            self.logp = tf.reduce_sum(softmax.loss * len_mask, axis=-1)
            self.loss = tf.reduce_sum(softmax.loss * len_mask) / tf.reduce_sum(len_mask)  # averaged over positions

        return outputs, tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    def compute_expectation(self, _outputs, _scalars):
        tvars = self.vars
        cost = tf.reduce_sum(_outputs * _scalars)
        grads = tf.gradients(cost, tvars)

        expectation = []   # a list of 'Tensor', to store the expectation of d\phi / d\theta
        clean_expec = []   # a list of 'operation' to set all the expectation to Zero
        update_expec = []  # a list of 'operation' to update the expectation
        for var, g in zip(tvars, grads):
            v = tf.Variable(np.zeros(var.shape, dtype='float32'), trainable=False)
            expectation.append(v)
            clean_expec.append(tf.assign(v, np.zeros(var.shape)))
            update_expec.append(tf.assign_add(v, g))

        return expectation, clean_expec, update_expec

    def compute_expectation2(self, _outputs, _scalars):
        tvars = self.vars
        cost = tf.reduce_sum(_outputs * tf.sqrt(_scalars))
        grads = tf.gradients(cost, tvars)

        expectation = []   # a list of 'Tensor', to store the expectation of d\phi / d\theta
        clean_expec = []   # a list of 'operation' to set all the expectation to Zero
        update_expec = []  # a list of 'operation' to update the expectation
        for var, g in zip(tvars, grads):
            v = tf.Variable(np.zeros(var.shape, dtype='float32'), trainable=False)
            expectation.append(v)
            clean_expec.append(tf.assign(v, np.zeros(var.shape)))
            g2 = tf.multiply(g, g)
            update_expec.append(tf.assign_add(v, g2))

        return expectation, clean_expec, update_expec

    def get_logz(self, session):
        logz_base = session.run(self._logz_base)
        zeta = np.append(np.zeros(self.config.min_len),
                         session.run(self._zeta) + logz_base)
        return zeta

    def get_zeta(self, session):
        return np.append(np.zeros(self.config.min_len), session.run(self._zeta))

    def set_pi(self, session, pi):
        return session.run(self._set_pi_0, {self._new_pi_or_zeta: pi})

    def set_zeta(self, session, zeta):
        return session.run(self._set_zeta, {self._new_pi_or_zeta: zeta})

    def set_logz_base(self, session, logz_base):
        return session.run(self._set_logz_base, {self._new_float: logz_base})

    def run_phi(self, session, inputs, lengths, extra_weight=None):
        if extra_weight is None:
            extra_weight = np.zeros(len(inputs), dtype='float32')
        return session.run(self._phi,
                           {self._inputs: inputs,
                            self._lengths: lengths,
                            self._extra_weight: extra_weight})

    def run_logps(self, session, inputs, lengths, extra_weight=None):
        if extra_weight is None:
            extra_weight = np.zeros(len(inputs), dtype='float32')
        return session.run(self._logp,
                           {self._inputs: inputs,
                            self._lengths: lengths,
                            self._extra_weight: extra_weight})

    def run_update_expec(self, session, inputs, lengths, scalars):
        if self.is_empirical:
            session.run([self._update_expec, self._update_expec2],
                        {self._inputs: inputs,
                         self._lengths: lengths,
                         self._scalars: scalars})
        else:
            session.run(self._update_expec,
                        {self._inputs: inputs,
                         self._lengths: lengths,
                         self._scalars: scalars})

    def run_clean_expec(self, session):
        if self.is_empirical:
            session.run([self.clean_expec, self.clean_expec2])
        else:
            session.run(self.clean_expec)

    def run_update_zeta(self, session, sample_pi, zeta_lr):
        session.run(self._update_zeta,
                    {self._sample_pi: sample_pi,
                     self._zeta_lr: zeta_lr})

    def run_summary(self, session, inputs, lengths):
        return session.run(self.summary_image,
                           {self._inputs: inputs,
                            self._lengths: lengths})

    def run_set_lr(self, session, lr):
        """[pretrain] set leraning rate"""
        assert 'train_layer' in self.__dict__
        self.train_layer.set_lr(session, lr)

    def run_train(self, session, inputs, lengths):
        """[pretrain] update parameters"""
        assert self.is_pretrain
        loss, _ = session.run([self.loss, self._train_op], {self._inputs: inputs, self._lengths: lengths})
        return loss

    def run_loss(self, session, inputs, lengths):
        assert self.is_pretrain
        return session.run(self.loss, {self._inputs: inputs, self._lengths: lengths})

    def run_logp(self, session, inputs, lengths):
        assert self.is_pretrain
        return session.run(self.logp, {self._inputs: inputs, self._lengths: lengths})


class TRF(trfbase.FastTRF):
    def __init__(self, config, data,
                 name='TRF', logdir='trf',
                 device='/gpu:0',
                 simulater_device=None,
                 network=Net):
        super().__init__(config, data, name=name, logdir=logdir, simulater_device=None)

        with tf.name_scope(name + '_Model'):
            self.model_net = network(config, is_training=True, is_empirical=False, device=device, reuse=None)
        with tf.name_scope(name + '_Empirical'):
            if config.opt_method.lower() == 'var':
                self.empir_net = network(config,
                                         is_training=True,
                                         is_empirical=True,
                                         device=device, reuse=True)
            else:
                self.empir_net = network(config,
                                         is_training=True,
                                         is_empirical=False,
                                         device=device, reuse=True)

        with tf.device(device), tf.name_scope(name):

            # update paramters
            grads = []
            for (es, et) in zip(self.model_net.expectation, self.empir_net.expectation):
                grads.append(es - et)

            self._lr_param = tf.Variable(0.001, trainable=False, name='learning_rate')  # learning rate
            self._global_step = tf.Variable(0, trainable=False, name='global_step')

            if config.opt_method.lower() == 'adam':
                optimizer = tf.train.AdamOptimizer(self._lr_param)
            elif config.opt_method.lower() == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self._lr_param)
            elif config.opt_method.lower() == 'var':
                beta = 0.9
                epsilon = config.var_gap
                # update empirical variance
                self.empirical_var = []
                self._update_empirical_var = []
                for exp, exp2 in zip(self.empir_net.expectation, self.empir_net.expectation2):
                    v = tf.Variable(np.ones(exp.shape, dtype='float32'), trainable=False)
                    self.empirical_var.append(v)
                    new_v = beta * v + (1 - beta) * (exp2 - exp * exp)
                    self._update_empirical_var.append(tf.assign(v, new_v))

                update_steps = [g / (var + epsilon) for g, var in zip(grads, self.empirical_var)]
                if config.max_grad_norm > 0:
                    grads, grads_norm = tf.clip_by_global_norm(update_steps, config.max_grad_norm)
                optimizer = tf.train.GradientDescentOptimizer(self._lr_param)

            else:
                if config.max_grad_norm > 0:
                    grads, grads_norm = tf.clip_by_global_norm(grads, config.max_grad_norm)
                optimizer = tf.train.GradientDescentOptimizer(self._lr_param)
            # optimizer = tf.train.GradientDescentOptimizer(self._lr_param)
            tvars = self.model_net.vars
            self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                       global_step=self._global_step)

            # update learining rate
            self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
            self._update_lr = tf.assign(self._lr_param, self._new_lr)

            # summarys:
            self._param_size = tf.add_n([tf.size(v) for v in self.model_net.vars])
            self._param_norm = tf.global_norm(self.model_net.vars)
            self._grad_norm = tf.global_norm(grads)

        # create simulater
        self.create_simulater(simulater_device=device if simulater_device is None else simulater_device)

        # summarys
        self._pure_summary = tf.summary.merge([
            tf.summary.scalar('TRF/param_size', self._param_size),
            tf.summary.scalar('TRF/param_norm', self._param_norm),
            tf.summary.scalar('TRF/grad_norm', self._grad_norm),
        ]
        )

        # saver
        self.saver = tf.train.Saver()
        self.is_load_model = False

    def phi(self, inputs, lengths):
        feat_weight = super().phi(inputs, lengths)  # weight of discrete features
        return self.model_net.run_phi(self.get_session(), inputs, lengths, feat_weight)

    def logps(self, inputs, lengths):
        feat_weight = super().phi(inputs, lengths)  # weight of discrete features
        return self.model_net.run_logps(self.get_session(), inputs, lengths, feat_weight)

    def get_log_probs(self, seq_list, is_norm=True):
        batch_size = self.config.batch_size
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

    def clean_expectation(self, session):
        self.empir_net.run_clean_expec(session)
        self.model_net.run_clean_expec(session)

    def true_normalize_all(self):
        super().true_normalize_all()
        self.model_net.set_logz_base(self.get_session(), self.true_normalize(self.config.min_len))
        self.model_net.set_zeta(self.get_session(), self.zeta)

    def update_zeta(self, sample_list, lr_zeta):
        # update zeta
        sample_pi = np.zeros(self.config.max_len+1)
        for seq in sample_list:
            sample_pi[len(seq)] += 1.
        sample_pi /= len(sample_list)

        self.model_net.run_update_zeta(self.get_session(), sample_pi, lr_zeta)
        self.model_net.set_logz_base(self.get_session(), self.true_normalize(self.config.min_len))

        self.sample_acc_count += sample_pi * len(sample_list)
        self.sample_cur_pi = sample_pi

        self.logz = self.model_net.get_logz(self.get_session())
        self.zeta = self.model_net.get_zeta(self.get_session())

    def update(self, session, train_seqs, sample_seqs, global_step):
        assert len(train_seqs) % self.config.batch_size == 0
        assert len(sample_seqs) % self.config.batch_size == 0

        lr_param = self.config.lr_param.get_lr(global_step)
        lr_zeta = self.config.lr_zeta.get_lr(global_step)
        lr_cnn = self.config.lr_cnn.get_lr(global_step)

        self.clean_expectation(session)

        def update_expec(net, seqs, s):
            for i in range(0, len(seqs), self.config.batch_size):
                x, n = reader.produce_data_to_trf(seqs[i: i + self.config.batch_size],
                                                  pad_value=self.config.end_token)
                net.run_update_expec(session, x, n, s[i: i + self.config.batch_size])

        train_scalars = 1.0 / len(train_seqs) * np.ones(len(train_seqs))
        update_expec(self.empir_net, train_seqs, train_scalars)

        sample_n = np.array([len(x) for x in sample_seqs])
        sample_scalars = 1.0 / len(sample_seqs) * self.config.pi_true[sample_n] / self.config.pi_0[sample_n]
        update_expec(self.model_net, sample_seqs, sample_scalars)

        # update params
        if self.config.opt_method.lower() == 'var':
            session.run(self._update_empirical_var)
        session.run(self._update_lr, {self._new_lr: lr_cnn})  # set lr
        session.run([self._train_op])

        # update zeta
        self.update_zeta(sample_seqs, lr_zeta)

        # update simulater
        self.simulater.update(self.get_session(), sample_seqs)

        # update discrete parameters
        if self.feat_word is not None:
            self.feat_word.seq_update(train_seqs, train_scalars, sample_seqs, sample_scalars,
                                      lr=lr_param,
                                      L2_reg=self.config.L2_reg,
                                      dropout=self.config.dropout)
        if self.feat_class is not None:
            self.feat_class.seq_update(self.data.seqs_to_class(train_seqs), train_scalars,
                                       self.data.seqs_to_class(sample_seqs), sample_scalars,
                                       lr=lr_param,
                                       L2_reg=self.config.L2_reg,
                                       dropout=self.config.dropout)

        return {'lr_cnn': lr_cnn, 'lr_param': lr_param, 'lr_zeta': lr_zeta}

    def eval(self, data_list):
        self.model_net.set_pi(self.get_session(), self.config.pi_true)
        logps = self.get_log_probs(data_list)
        self.model_net.set_pi(self.get_session(), self.config.pi_0)

        lens = [len(x) - int(self.config.beg_token is not None) for x in data_list]
        s = - sum(logps)
        nll = s / len(data_list)
        ppl = np.exp(s/sum(lens))
        return nll, ppl

    def eval_pi0(self, data_list):
        logps = self.get_log_probs(data_list)

        lens = [len(x) - int(self.config.beg_token is not None) for x in data_list]
        s = - sum(logps)
        nll = s / len(data_list)
        ppl = np.exp(s/sum(lens))
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
        print('[TRF] nn.param_num={:,}'.format(self.get_session().run(self._param_size)))
        if not self.is_load_model and self.config.pretrain_net_path is not None:
            print('[TRF] load pretrain parameters...')
            self.model_net.pretrain_saver.restore(self.get_session(), self.config.pretrain_net_path)

    def train_after_update(self, **argv):
        eval_list = argv['eval_list']
        sv = argv['sv']
        session = argv['session']

        if eval_list is not None:
            summ = self.model_net.run_summary(session, *reader.produce_data_to_trf(eval_list))
            sv.summary_computed(session, summ)


def main(_):
    data = reader.Data().load_raw_data(reader.word_raw_dir(),
                                       add_beg_token='<s>', add_end_token='</s>',
                                       add_unknwon_token=None,
                                       max_length=5)

    config = trfbase.Config(data)
    config.embedding_dim = 128
    # config.load_embedding_path = './embedding/emb_{}x{}.ckpt'.format(config.vocab_size, config.embedding_dim)
    config.pprint()

    # wb.rmdir(logdirs)
    with tf.Graph().as_default():
        m = TRF(config, data, logdir='../egs/word/test')

        sv = tf.train.Supervisor(logdir='../egs/word/test/logs', summary_op=None, global_step=m._global_step)
        sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs

        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:
            m.set_session(session)

            m.test_sample()

            # train_seqs = data.datas[0][0: config.batch_size]
            # sample_seqs = train_seqs
            # _, summ = m.update(session, train_seqs, sample_seqs)
            #
            # sv.summary_computed(session, summ)

            # a = data.datas[0][0: 1]
            # print(a)
            # print(session.run(m.model_net._pi_0))
            # print(m.model_net.get_logz(session))
            # print(m.logps(*reader.produce_data_to_trf(a)))
            # print(m.phi(*reader.produce_data_to_trf(a)))

            # s = ['organiation', 'application', 'applicances', 'banana']
            # eval_list = data.load_data([[data.beg_token_str] + list(w) + [data.end_token_str] for w in s])
            # print(eval_list)
            #
            # m.train(session, sv,
            #         print_per_epoch=0.1,
            #         eval_list=eval_list)


if __name__ == '__main__':
    tf.app.run(main=main)
