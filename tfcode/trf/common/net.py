import tensorflow as tf

from base import *


class Config(wb.Config):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.structure_type = 'cnn'  # 'cnn' or 'rnn' or 'mix'
        self.embedding_dim = 128
        self.load_embedding_path = None
        # cnn structure
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
        self.init_weight = 0.1
        # self.max_grad_norm = 10
        # self.opt_method = 'sgd'

    def __str__(self):
        s = 'e{}'.format(self.embedding_dim)
        # cnn structure
        if self.structure_type == 'cnn' or self.structure_type == 'mix':
            s += '_cnn'
            if self.cnn_batch_normalize:
                s += '_BN'
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

        return s


def create_net(config, is_training, reuse):
    if config.structure_type == 'cnn':
        return NetCNN(config, is_training, reuse)
    elif config.structure_type == 'rnn':
        return NetRnn(config, is_training, reuse)
    elif config.structure_type == 'mix':
        return NetMix(config, is_training, reuse)
    else:
        raise TypeError('Undefined net type={}'.format(config.structure_type))


class NetBase(object):
    def __init__(self, config, is_training, reuse=None):
        self.config = config
        self.is_training = is_training
        self.reuse = reuse

    def output(self, _inputs, _lengths, reuse=None):
        """
        given the inputs and lengths, output the phis of each sentences and the variables in the networks
        Args:
            _inputs: a Tensor
            _lengths: a Tensor
            reuse: None or True

        Returns:
            tuple( a Tensor indicates the phis,
                   a variable list )
        """
        pass


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

    def build_cnn(self, _inputs, _lengths, reuse=None):
        """
                Using the self._inputs and self._lengths to calculate phi of TRF
                """
        vocab_size = self.config.vocab_size
        embedding_dim = self.config.embedding_dim

        batch_size = tf.shape(_inputs)[0]
        max_len = tf.shape(_inputs)[1]
        # the length mask, the position < len is 1., otherwise is 0.
        len_mask = tf.tile(tf.reshape(tf.range(max_len, dtype=tf.int32), [1, max_len]), [batch_size, 1])
        len_mask = tf.less(len_mask, tf.reshape(_lengths, [batch_size, 1]))
        len_mask = tf.cast(len_mask, tf.float32)  # shape (batch_size, max_len)
        expand_len_mask = tf.expand_dims(len_mask, axis=-1)  # shape (batch_size, max_len, 1)
        tf.summary.image('len_mask', tf.expand_dims(expand_len_mask, axis=0), collections=['cnn'])

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
                                             trainable=True)
        else:
            word_embedding = tf.get_variable('word_embedding',
                                             [vocab_size, embedding_dim], dtype=tf.float32)
        emb_output = tf.nn.embedding_lookup(word_embedding, _inputs)  # (batch_size, seq_len, emb_dim)

        # dropout
        if self.is_training and self.config.dropout > 0:
            emb_output = tf.nn.dropout(emb_output, keep_prob=1. - self.config.dropout)

        # pre-net
        emb_output = layers.linear(emb_output, self.config.cnn_hidden, tf.nn.relu, name='pre_net1')
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
        for (filter_width, out_dim) in self.config.cnn_filters:
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
        for i in range(self.config.cnn_layers):
            conv_output = tf.layers.conv1d(
                inputs=inputs,
                filters=self.config.cnn_hidden,
                kernel_size=self.config.cnn_width,
                padding='same',
                activation=None,
                reuse=reuse,
                name='cnn{}'.format(i + 1)
            )
            conv_output = self.activation(conv_output, reuse, 'cnn{}/BN'.format(i + 1))
            conv_output = conv_output * expand_len_mask
            tf.summary.image('cnn{}'.format(i + 1),
                             tf.expand_dims(conv_output, axis=-1),
                             max_outputs=4, collections=['cnn'])

            if self.config.cnn_skip_connection:
                skip_scalar = tf.get_variable(name='cnn{}_skip_scalar'.format(i + 1),
                                              shape=[self.config.cnn_hidden], dtype=tf.float32)
                skip_connections.append(conv_output * tf.reshape(skip_scalar, [1, 1, self.config.cnn_hidden]))

            inputs = conv_output

        # skip connections
        if skip_connections:
            inputs = self.activation(tf.add_n(skip_connections), reuse, 'skip_conn/BN')

        # residual connection
        if self.config.cnn_residual:
            inputs = tf.nn.relu(emb_output + inputs)

        tf.summary.image('cnn_end',
                         tf.expand_dims(inputs, axis=-1),
                         max_outputs=4, collections=['cnn'])

        return inputs

    def output(self, _inputs, _lengths, reuse=None):
        inputs = self.build_cnn(_inputs, _lengths, reuse)

        batch_size = tf.shape(_inputs)[0]
        max_len = tf.shape(_inputs)[1]
        len_mask = tf.sequence_mask(_lengths, maxlen=max_len, dtype=tf.float32)

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

        return outputs, \
               tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)


class NetRnn(NetBase):
    def __init__(self, config, is_training, reuse=None):

        self._forward_init_state = None
        self._forward_final_state = None
        self.fstate = None
        self.fstate2 = None

        super().__init__(config, is_training, reuse)

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
            self._forward_init_state = cell_fw.zero_state(batch_size, tf.float32)
            outputs, states = tf.nn.dynamic_rnn(cell_fw,
                                                inputs=inputs,
                                                sequence_length=_lengths - 1,
                                                initial_state=self._forward_init_state)
            outputs_fw = outputs
            outputs_bw = None
            self._forward_final_state = states

        return outputs_fw, outputs_bw, states, emb

    def output(self, _inputs, _lengths, reuse=None):

        outputs_fw, outputs_bw, _, emb = self.compute_rnn(_inputs, _lengths, reuse)

        batch_size = tf.shape(_inputs)[0]

        if not self.config.rnn_predict:
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

    def feed_state(self, to_state_tensor, state):
        """
        return feed dictionary used in session.run,
        to set the initial state

        Args:
            state: the lstm state, a tuple of tuples

        Returns:
            a feed dict
        """
        feed_dict = {}
        for (_c, _h), (c, h) in zip(to_state_tensor, state):
            feed_dict[_c] = c
            feed_dict[_h] = h
        return feed_dict

    # def get_phi(self, seq_list):
    #     inputs, lengths = reader.produce_data_to_array(seq_list)
    #     feed_dict = {self._inputs: inputs,
    #                  self._lengths: lengths}
    #
    #     if self.fstate is not None:
    #         feed_dict.update(self.feed_state(self._forward_init_state, self.fstate))
    #
    #     phi, self.fstate2 = tf.get_default_session().run([self.phi, self._forward_final_state], feed_dict)
    #     return phi
    #
    # def update(self, seq_list, cluster_weights, cluster_m=None, learning_rate=1.0):
    #     inputs, lengths = reader.produce_data_to_array(seq_list)
    #     feed_dict = {self._inputs: inputs,
    #                  self._lengths: lengths,
    #                  self._cluster_weights: cluster_weights}
    #
    #     # set learning rate
    #     self.trainop.set_lr(tf.get_default_session(), learning_rate)
    #
    #     # update parameters
    #     if self.fstate is not None:
    #         feed_dict.update(self.feed_state(self._forward_init_state, self.fstate))
    #
    #     self.trainop.update(tf.get_default_session(), feed_dict)


class NetMix(NetCNN, NetRnn):

    def output(self, _inputs, _lengths, reuse=None):

        # CNN
        inputs = NetCNN.build_cnn(_inputs, _lengths, reuse)

        # LSTM
        outputs_fw, outputs_bw, _, _ = NetRnn.compute_rnn(inputs, _lengths, reuse)
        inputs = tf.concat([outputs_fw, outputs_bw], axis=2)

        if self.config.attention:
            attention_weight = layers.linear(inputs, 1, activate=tf.nn.sigmoid, name='attention_weight')
            # summate
            inputs *= attention_weight

        # final layers
        len_mask = tf.sequence_mask(_lengths, maxlen=tf.shape(_inputs)[1], dtype=tf.float32)
        expand_len_mask = tf.expand_dims(len_mask, axis=-1)
        outputs = tf.reduce_sum(inputs * expand_len_mask, axis=1)  # [batch_size, dim]
        outputs = layers.linear(outputs, 1, name='final_linear')  # [batch_size, 1]
        outputs = tf.squeeze(outputs, axis=[-1])  # [batch_size]

        return outputs, tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
