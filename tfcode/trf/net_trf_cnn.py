import tensorflow as tf
import numpy as np

from . import reader
from . import layers
from . import word2vec
from . import wblib as wb


def get_activation(name):
    # define the activation
    if name == 'none':
        act = None
    elif name == 'relu':
        act = tf.nn.relu
    elif name == 'linear':
        a = tf.get_variable('linear_act_a', [], dtype=tf.float32)
        b = tf.get_variable('linear_act_b', [], dtype=tf.float32)
        act = lambda x: a * x + b
    elif name == 'norm':
        act = lambda x: x / (1e-12 + tf.norm(x, axis=-1, keep_dims=True))
    else:
        raise TypeError('undefined activation = ' + name)
    return act


class NetTRF(object):
    class Config(wb.PPrintObj):
        def __init__(self, data):
            self.min_len = data.get_min_len()
            self.max_len = data.get_max_len()
            self.vocab_size = data.get_vocab_size()
            self.beg_token = data.get_beg_token()
            self.end_token = data.get_end_token()
            self.pi_true = data.get_pi_true()
            self.pi_0 = data.get_pi0(self.pi_true)

            # for structure
            self.embedding_dim = 128
            self.load_embedding_path = None
            self.update_embedding = True
            self.cnn_banks = [(i, 128) for i in range(1, 6)]
            self.cnn_stacks = [(3, 128), (3, 138), (3, 128)]
            self.cnn_activation = 'relu'  # one of ['relu', 'norm', 'linear']
            self.cnn_use_bias = True
            self.cnn_skip_connection = True
            self.cnn_residual = False
            self.init_weight = 0.1

            # for training
            self.opt_method = 'adam'
            self.train_batch_size = 100
            self.sample_batch_size = 100
            self.update_batch_size = 100  # used in run_training, to constraint memory usage
            self.zeta_gap = 10
            self.max_grad_norm = 10.0
            self.dropout = 0.0

        def __str__(self):
            s = 'cnn_'
            s += 'e{}'.format(self.embedding_dim)
            if self.cnn_banks:
                a = list(map(lambda x: x[1] == self.cnn_banks[0][1], self.cnn_banks))
                if all(a):
                    s += '_({}to{}x{})'.format(self.cnn_banks[0][0],
                                                 self.cnn_banks[-1][0],
                                                 self.cnn_banks[0][1])
                else:
                    s += '_(' + '_'.join(['{}x{}'.format(w, d) for (w, d) in self.cnn_banks]) + ')'

            if self.cnn_stacks:
                s += '_(' + '_'.join(['{}x{}'.format(w, d) for (w, d) in self.cnn_stacks]) + ')'
            s += '_' + self.cnn_activation
            return s

        def initial_zeta(self):
            len_num = self.max_len - self.min_len + 1
            logz = np.append(np.zeros(self.min_len),
                             np.log(self.vocab_size) * np.linspace(1, len_num, len_num))
            zeta = logz - logz[self.min_len]
            return zeta

    def __init__(self, config, reuse=None, is_training=True, name='net_trf', propose_lstm=None):
        self.config = config
        self.is_training = is_training
        self.name = name
        self.default_initializer = tf.random_uniform_initializer(-self.config.init_weight, self.config.init_weight)
        # self.default_initializer = tf.random_normal_initializer(0.0, self.config.init_weight)
        is_training = True

        # create pi/zeta/zeta_base variables
        with tf.variable_scope(self.name, reuse=reuse, initializer=self.default_initializer):
            valid_len = config.max_len - config.min_len + 1
            self.pi = tf.get_variable('pi', shape=[valid_len], dtype=tf.float32,
                                      trainable=False,
                                      initializer=tf.constant_initializer(config.pi_0[config.min_len:]))
            self.zeta = tf.get_variable('zeta', shape=[valid_len], dtype=tf.float32,
                                        trainable=False,
                                        initializer=tf.constant_initializer(config.initial_zeta()[config.min_len:]))
            self.logz_base = tf.get_variable('logz_base', shape=[], dtype=tf.float32, trainable=False,
                                             initializer=tf.constant_initializer(np.log(self.config.vocab_size)))
        # create all variables
        # inputs: of shape (batch_size, seq_len)
        # lengths: of shape (batch_size,)
        with tf.name_scope(self.name):
            self._inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
            self._lengths = tf.placeholder(tf.int32, [None], name='lengths')
            self.dbgs = None
            self.phi = self.get_phi(self._inputs, self._lengths, reuse=reuse)
            self.logps = self.get_logp(self._inputs, self._lengths)

            # variables
            self.vars = self.get_vars()
            if reuse is None:
                print('variables in %s' % self.name)
                for v in self.vars:
                    print('\t' + v.name, v.shape, v.device)

            # set pi and zeta
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
            self.grads_to_params = self.compute_gradient_to_params(self.phi[0: self._train_num],
                                                                   self.phi[self._train_num:],
                                                                   self._lengths[self._train_num:])
            self.grads_to_zeta = self.compute_gradient_to_zeta(self._lengths[self._train_num:])

            #############################################################
            # update parameters
            #############################################################
            if is_training and not reuse:
                self._lr = [tf.Variable(1.0, trainable=False, name='learning_rate_param'),
                            tf.Variable(1.0, trainable=False, name='learning_rate_zeta')]
                self.global_step = tf.Variable(0, trainable=False, name='global_step')

                # save the gradient to memory
                self.grads_vars = []  # variables to save the gradients
                self.grads_vars_clean = []  # ops to zero the gradient buffer
                self.grads_vars_add = []  # ops to add the gradients
                for g, var in zip(self.grads_to_params, self.vars):
                    v = tf.Variable(np.zeros(var.shape), dtype=tf.float32, trainable=False)
                    self.grads_vars.append(v)
                    self.grads_vars_clean.append(tf.assign(v, tf.zeros_like(v)))
                    self.grads_vars_add.append(tf.assign_add(v, g))

                # optimizer
                if config.opt_method.lower() == 'adam':
                    optimizer = tf.train.AdamOptimizer(self._lr[0])
                elif config.opt_method.lower() == 'adagrad':
                    optimizer = tf.train.AdagradOptimizer(self._lr[0])
                elif config.opt_method.lower() == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(self._lr[0])
                else:
                    raise TypeError('undefined opt method = ' + config.opt_method)

                if config.max_grad_norm is not None:
                    grads, _ = tf.clip_by_global_norm(self.grads_vars, config.max_grad_norm)
                else:
                    grads = self.grads_vars

                # update variables
                self.update_params = optimizer.apply_gradients(zip(grads, self.vars),
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

    def compute_phi(self, _inputs, _lengths, reuse=True):
        """reuse all the variables to compute the phi"""
        vocab_size = self.config.vocab_size
        embedding_dim = self.config.embedding_dim
        dbgs = []

        # define the activation
        # act = get_activation(self.config.cnn_activation)

        def create_cnn(inputs, dim, width, name):
            if self.config.cnn_activation == 'gated':
                o = tf.layers.conv1d(inputs=inputs,
                                     filters=dim,
                                     kernel_size=width,
                                     padding='same',
                                     use_bias=self.config.cnn_use_bias,
                                     activation=None,
                                     reuse=reuse,
                                     name=name + '_o')
                g = tf.layers.conv1d(inputs=inputs,
                                     filters=dim,
                                     kernel_size=width,
                                     padding='same',
                                     use_bias=self.config.cnn_use_bias,
                                     activation=None,
                                     reuse=reuse,
                                     name=name + '_g')
                outputs = tf.nn.tanh(o) * tf.nn.sigmoid(g)
            else:
                act = get_activation(self.config.cnn_activation)
                outputs = tf.layers.conv1d(inputs=inputs,
                                           filters=dim,
                                           kernel_size=width,
                                           padding='same',
                                           use_bias=self.config.cnn_use_bias,
                                           activation=act,
                                           reuse=reuse,
                                           name=name)
            return outputs

        batch_size = tf.shape(_inputs)[0]
        max_len = tf.shape(_inputs)[1]
        # the length mask, the position < len is 1., otherwise is 0.
        len_mask, expand_len_mask = layers.generate_length_mask(_lengths, batch_size, max_len)

        # embedding layers
        if self.config.load_embedding_path is not None:
            print('read init embedding vector from', self.config.load_embedding_path)
            emb_init_value = word2vec.read_vec(self.config.load_embedding_path)
            if emb_init_value.shape != (vocab_size, embedding_dim):
                raise TypeError('the reading embedding with shape ' +
                                str(emb_init_value.shape) +
                                ' does not match current shape ' +
                                str([vocab_size, embedding_dim]) +
                                '\nform path ' + self.config.load_embedding_path)
            word_embedding = tf.get_variable('word_embedding',
                                             [vocab_size, embedding_dim], dtype=tf.float32,
                                             initializer=tf.constant_initializer(emb_init_value),
                                             trainable=self.config.update_embedding)
        else:
            word_embedding = tf.get_variable('word_embedding',
                                             [vocab_size, embedding_dim], dtype=tf.float32)
        emb_output = tf.nn.embedding_lookup(word_embedding, _inputs)  # (batch_size, seq_len, emb_dim)
        # emb_output = tf.one_hot(_inputs, self.config.vocab_size)

        dbgs.append(emb_output)
        # dropout
        if self.config.dropout > 0:
            emb_output = tf.nn.dropout(emb_output, keep_prob=1. - self.config.dropout)

        # pre-net
        # emb_output = layers.linear(emb_output, self.config.cnn_hidden, activate=act, name='pre_net')

        # dropout
        # if self.config.dropout > 0:
        #     emb_output = tf.nn.dropout(emb_output, keep_prob=1. - self.config.dropout)

        inputs = emb_output * expand_len_mask
        # cnn layer-0
        # compute cnn with different filter width
        conv_list = []
        for (filter_width, out_dim) in self.config.cnn_banks:
            # cnn output is of shape (bacth_size, seq_len, out_dim)
            conv_output = create_cnn(inputs, out_dim, filter_width, name='cnn0_%d' % filter_width)
            conv_list.append(conv_output * expand_len_mask)

        if conv_list:
            inputs = tf.concat(conv_list, axis=-1)

        # max_pooling
        inputs = tf.layers.max_pooling1d(inputs, pool_size=2, strides=1, padding='same')
        inputs *= expand_len_mask
        # tf.summary.image('cnn0_pooling', tf.expand_dims(inputs, axis=-1), max_outputs=4, collections=['cnn'])

        # several cnn layers
        skip_connections = []
        for i, (filter_width, out_dim) in enumerate(self.config.cnn_stacks):
            conv_output = create_cnn(inputs, out_dim, filter_width, 'cnn%d' % (i+1))
            conv_output = conv_output * expand_len_mask

            if self.config.cnn_skip_connection:
                skip_connections.append(conv_output)

            inputs = conv_output

        # skip connections
        if skip_connections:
            inputs = tf.concat(skip_connections, axis=-1)

        # final conv
        # defferent weights for different lengths
        # weights = tf.get_variable('weights', [self.config.max_len - self.config.min_len + 1, inputs.shape[-1].value])
        # cur_w = tf.gather(weights, _lengths - self.config.min_len)  # [batch_size, dims]
        # outputs = tf.reduce_sum(inputs * tf.expand_dims(cur_w, axis=1), axis=-1)  # [batch_size, max_len]

        outputs = layers.linear(inputs, 1, name='final_layers')
        outputs = tf.reshape(outputs, [batch_size, -1])

        outputs = outputs * len_mask
        outputs = tf.reduce_sum(outputs, axis=-1)  # of shape [batch_size]

        if self.dbgs is None:
            self.dbgs = dbgs

        return outputs

    def get_vars(self):
        with tf.variable_scope(self.name, reuse=True):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_phi(self, _inputs, _lengths, reuse=True):
        """wrapped the comput_phi into a variable scorp"""
        with tf.variable_scope(self.name, reuse=reuse, initializer=self.default_initializer):
            outputs = self.compute_phi(_inputs, _lengths, reuse)
        return outputs

    def get_logp(self, _inputs, _lengths, reuse=True, name='trf_logp'):
        with tf.name_scope(name):
            pi = tf.log(tf.gather(self.pi, _lengths - self.config.min_len))
            zeta = tf.gather(self.zeta + self.logz_base, _lengths - self.config.min_len)
            return self.get_phi(_inputs, _lengths, reuse=reuse) + pi - zeta

    def compute_gradient_to_params(self, train_phi, sample_phi, sample_lengths, name='grad_to_params'):
        with tf.name_scope(name):
            train_loss = tf.reduce_sum(train_phi) / self.config.train_batch_size
            # train_grads = tf.gradients(train_loss, self.vars)

            pi_rate = self.config.pi_true[self.config.min_len:] / self.config.pi_0[self.config.min_len:]
            weights = tf.gather(pi_rate.astype('float32'), sample_lengths - self.config.min_len)
            sample_loss = tf.reduce_sum(sample_phi * weights) / self.config.sample_batch_size
            # sample_grads = tf.gradients(sample_loss, self.vars)

            grads = tf.gradients(sample_loss - train_loss, self.vars)

            return grads

    def compute_gradient_to_params_loop(self, train_phi, sample_phi, sample_lengths, iter_num=10, name='grad_to_params_loop'):
        with tf.name_scope(name):
            assert self.config.train_batch_size % iter_num == 0
            assert self.config.sample_batch_size % iter_num == 0
            print('loop_grads iter_num=', iter_num)

            train_step = self.config.train_batch_size // iter_num
            sample_step = self.config.sample_batch_size // iter_num
            pi_rate = self.config.pi_true[self.config.min_len:] / self.config.pi_0[self.config.min_len:]

            def get_g(i):
                train_loss = tf.reduce_sum(
                    train_phi[i * train_step: (i + 1) * train_step]) / self.config.train_batch_size

                weights = tf.gather(pi_rate.astype('float32'),
                                    sample_lengths[i * sample_step: (i + 1) * sample_step] - self.config.min_len)
                sample_loss = tf.reduce_sum(
                    sample_phi[i * sample_step: (i + 1) * sample_step] * weights) / self.config.sample_batch_size

                cur_gs = tf.gradients(sample_loss - train_loss, self.vars)
                return cur_gs

            def body(i, gs):
                cur_gs = get_g(i)
                new_gs = []
                for g, cur_g in zip(gs, cur_gs):
                    g = tf.expand_dims(g, axis=0)
                    cur_g = tf.expand_dims(cur_g, axis=0)
                    new_gs.append(tf.reduce_sum(tf.concat([g, cur_g], axis=0), axis=0))

                return i+1, new_gs

            def cond(i, gs):
                return tf.less(i, iter_num)

            initial_grads = get_g(0)
            _, grads = tf.while_loop(cond, body, loop_vars=[1, initial_grads],
                                     parallel_iterations=1)
            return grads

    def compute_gradient_to_zeta(self, sample_lengths, name='grad_to_zeta'):
        with tf.name_scope(name):
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

    def run_phi(self, session, inputs, lengths):
        return session.run(self.phi,
                           {self._inputs: inputs,
                            self._lengths: lengths})

    def run_logps(self, session, inputs, lengths):
        return session.run(self.logps,
                           {self._inputs: inputs,
                            self._lengths: lengths})

    def run(self, session, inputs, lengths, train_num, ops):
        return session.run(ops,
                           {self._inputs: inputs,
                            self._lengths: lengths,
                            self._train_num: train_num})

    def run_train(self, session, inputs, lengths, train_num):
        assert train_num == self.config.train_batch_size
        assert len(inputs) - train_num == self.config.sample_batch_size
        # update parameters
        # clean the buffer
        session.run(self.grads_vars_clean)
        # batch calculate the gradient
        batch_num = train_num // self.config.update_batch_size
        train_batch_size = train_num // batch_num
        sample_batch_size = (len(inputs)-train_num) // batch_num
        seqs = reader.extract_data_from_trf(inputs, lengths)
        train_seqs = seqs[0: train_num]
        sample_seqs = seqs[train_num:]
        for i in range(batch_num):
            cur_train = train_seqs[i * train_batch_size: (i+1) * train_batch_size]
            cur_sample = sample_seqs[i * sample_batch_size: (i+1) * sample_batch_size]

            cur_x, cur_n = reader.produce_data_to_trf(cur_train + cur_sample)
            self.run(session, cur_x, cur_n, len(cur_train), ops=self.grads_vars_add)
        # update the parameters
        session.run(self.update_params)

        # update zeta
        session.run(self.update_zeta, {self._lengths: lengths, self._train_num: train_num})

    def run_set_lr(self, session, lr1, lr2):
        session.run(self._set_lr[0], {self._new_lr: lr1})
        session.run(self._set_lr[1], {self._new_lr: lr2})
