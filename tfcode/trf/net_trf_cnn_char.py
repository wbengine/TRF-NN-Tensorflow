import tensorflow as tf
import numpy as np

from . import net_trf_cnn as trf
from . import layers
from . import word2vec
from . import reader
from . import wblib as wb


class NetTRF(trf.NetTRF):
    class Config(trf.NetTRF.Config):
        def __init__(self, data):
            super().__init__(data)
            self.waist_dim = None
            # for char structure
            self.word_to_char_id = data.word_to_chars  # a list of list
            self.char_vocab_size = data.get_char_size()
            self.char_embedding_size = 15
            self.char_nn_type = 'cnn'  # one of 'lstm', 'cnn'
            self.char_cnn_bank = [(i, min(200, 50 * i)) for i in range(1, 8)]  # if char_nn_type='cnn'

        def __str__(self):
            return 'char' + super().__str__()

        def initial_zeta(self):
            len_num = self.max_len - self.min_len + 1
            logz = np.append(np.zeros(self.min_len),
                             np.log(self.vocab_size) * np.linspace(1, len_num, len_num))
            zeta = logz - logz[self.min_len]
            return zeta

    def __init__(self, config, reuse=None, is_training=True, name='net_trf',  propose_lstm=None):
        with tf.variable_scope(name, reuse=reuse):
            with tf.device('/gpu:0'):
                word2chars_arys, word2char_lens = reader.produce_data_to_trf(config.word_to_char_id)
                self.w2c_arys = tf.constant(word2chars_arys, dtype=tf.int32, name='w2c_arys')
                self.w2c_lens = tf.constant(word2char_lens, dtype=tf.int32, name='w2c_lens')
        super().__init__(config, reuse=reuse, is_training=is_training, name=name)

    def compute_embedding_rnn(self, _inputs, _lengths, reuse):
        with tf.variable_scope('char_net', reuse=reuse):
            # _inputs: [batch_size, max_len]
            # _lengths: [batch_size]
            batch_size = tf.shape(_inputs)[0]
            max_len = tf.shape(_inputs)[1]
            # [batch_size * max_len]
            words = tf.reshape(_inputs, [-1])
            # [batch_size * max_len]
            chars_len = tf.gather(self.w2c_lens, words, name='word2chars_len')
            # [batch_size * max_len, max_char]
            chars_id = tf.gather(self.w2c_arys, words, name='word2chars_id')
            chars_id = chars_id[:, 0: tf.reduce_max(chars_len)]  # reduce the space

            # character embedding
            char_embedding = tf.get_variable('char_embedding',
                                             [self.config.char_vocab_size, self.config.char_embedding_size],
                                             dtype=tf.float32)
            # [batch_size * max_len, max_char, char_emb]
            char_inputs = tf.nn.embedding_lookup(char_embedding, chars_id)

            # define the lstm of each chars
            def char_lstm_cell():
                c = tf.contrib.rnn.BasicLSTMCell(self.config.char_embedding_size, forget_bias=0., reuse=reuse)
                return c
            cell_fw = tf.contrib.rnn.MultiRNNCell([char_lstm_cell() for _ in range(1)])
            cell_bw = tf.contrib.rnn.MultiRNNCell([char_lstm_cell() for _ in range(1)])

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                              inputs=char_inputs,
                                                              sequence_length=chars_len,
                                                              dtype=tf.float32)

            # get the last hidden outputs and stack together
            tail_index = tf.stack([tf.range(tf.shape(chars_len)[0]), chars_len-1], axis=1)
            fw_output = tf.gather_nd(outputs[0], tail_index)
            bw_output = outputs[1][:, 0, :]
            # [batch_size * max_len, 1, char_emb * 2]
            outputs = tf.concat([fw_output, bw_output], axis=-1)
            # outputs = tf.reshape(outputs, [-1, outputs.shape[-1].value])

            # [batch_size, max_len, sum_of_cnn_dim]
            word_emb = tf.reshape(outputs, [batch_size, max_len, self.config.char_embedding_size * 2])

            # [batch_size, max_len, sum_of_cnn_dim]
            # word_emb *= tf.expand_dims(tf.sequence_mask(_lengths, tf.shape(_inputs)[1], dtype=tf.float32), axis=-1)

            return word_emb

    def compute_embedding_cnn(self, _inputs, _lengths, reuse):
        """compute the word embedding"""
        # _inputs: [batch_size, max_len]
        # _lengths: [batch_size]
        batch_size = tf.shape(_inputs)[0]
        max_len = tf.shape(_inputs)[1]
        # [batch_size * max_len]
        words = tf.reshape(_inputs, [-1])
        # [batch_size * max_len]
        chars_len = tf.gather(self.w2c_lens, words, name='word2chars_len')
        # [batch_size * max_len, max_char]
        chars_id = tf.nn.embedding_lookup(self.w2c_arys, words, name='word2chars_id')
        chars_id = chars_id[:, 0: tf.reduce_max(chars_len)]  # reduce the space

        # [batch_size * max_len, max_char]
        char_len_mask = tf.sequence_mask(chars_len, maxlen=tf.shape(chars_id)[1], dtype=tf.float32)
        char_len_mask = tf.expand_dims(char_len_mask, axis=-1)

        # character embedding
        char_embedding = tf.get_variable('char_embedding',
                                         [self.config.char_vocab_size, self.config.char_embedding_size],
                                         dtype=tf.float32)
        # [batch_size * max_len, max_char, char_emb]
        char_inputs = tf.nn.embedding_lookup(char_embedding, chars_id)
        char_inputs *= char_len_mask

        # perform cnn
        cnn_outputs = []
        cnn_output_dim = sum([d for w, d in self.config.char_cnn_bank])
        for width, filters in self.config.char_cnn_bank:
            # [batch_size * max_len, max_char, filters]
            outputs = tf.layers.conv1d(inputs=char_inputs,
                                       filters=filters,
                                       kernel_size=width,
                                       padding='same',
                                       activation=tf.nn.relu,
                                       use_bias=True,
                                       name='char_cnn_%d' % width,
                                       reuse=reuse)
            # [batch_size * max_len, filters]
            outputs = tf.reduce_max(outputs, axis=1)
            cnn_outputs.append(outputs)

        # [batch_size * max_len, sum_of_cnn_dim]
        outputs = tf.concat(cnn_outputs, axis=-1)

        # [batch_size, max_len, sum_of_cnn_dim]
        word_emb = tf.reshape(outputs, [batch_size, max_len, cnn_output_dim])

        # [batch_size, max_len, word_embedding]
        word_emb = layers.linear(word_emb, self.config.embedding_dim,
                                 activate=tf.nn.relu, name='word_emb_layers')

        return word_emb

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
                act = trf.get_activation(self.config.cnn_activation)
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
        emb_output = self.compute_embedding_cnn(_inputs, _lengths, reuse)  # (batch_size, seq_len, emb_dim)
        # emb_output = tf.one_hot(_inputs, self.config.vocab_size)
        dbgs.append(emb_output)

        # pre-net
        # emb_output = layers.linear(emb_output, self.config.cnn_hidden, activate=act, name='pre_net')

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

        # waist layers
        if self.config.waist_dim is not None:
            inputs = layers.linear(inputs, self.config.waist_dim, activate=tf.nn.relu, name='waist_layers')

        # several cnn layers
        skip_connections = []
        for i, (filter_width, out_dim) in enumerate(self.config.cnn_stacks):
            conv_output = create_cnn(inputs, out_dim, filter_width, 'cnn%d' % (i + 1))
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



