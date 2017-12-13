import tensorflow as tf
import numpy as np

from . import net_trf_rnn as trf
from . import layers
from . import word2vec
from . import reader
from . import wblib as wb


class NetTRF(trf.NetTRF):
    class Config(trf.NetTRF.Config):
        def __init__(self, data):
            super().__init__(data)
            # for char structure
            self.word_to_char_id = data.word_to_chars  # a list of list
            self.char_vocab_size = data.get_char_size()
            self.char_embedding_size = 100
            self.char_nn_type = 'lstm'  # one of 'lstm', 'cnn'
            self.char_cnn_bank = [(i, min(200, 50 * i)) for i in range(1, 8)]  # if char_nn_type='cnn'

        def __str__(self):
            return 'char_{}_e{}_word_b{}_e{}_h{}x{}'.format(self.char_nn_type, self.char_vocab_size,
                                                            self.type, self.embedding_dim,
                                                            self.hidden_dim, self.hidden_layers)

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
        super().__init__(config, reuse=reuse, is_training=is_training, name=name, propose_lstm=propose_lstm)

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
        # chars_id = chars_id[:, 0: tf.reduce_max(chars_len)]  # reduce the space

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
                                       padding='valid',
                                       activation=tf.tanh,
                                       use_bias=True,
                                       name='char_cnn_%d' % width,
                                       reuse=reuse)
            # [batch_size * max_len, filters]
            outputs = tf.reduce_max(outputs, axis=1)
            cnn_outputs.append(outputs)

        # [batch_size * max_len, sum_of_cnn_dim]
        outputs = tf.concat(cnn_outputs, axis=-1)
        # outputs *= char_len_mask
        #
        # # compute max-pool over times
        # # min value at each dimension, [batch_size * max_len, 1, hidden_size]
        # min_value = tf.reduce_min(outputs, axis=1, keep_dims=True)
        # # [batch_size * max_len, max_char, hidden_size]
        # min_value = tf.tile(min_value, [1, tf.shape(outputs)[1], 1])
        # # pad to the min_value out of the length
        # outputs = outputs * char_len_mask + min_value * (1.0 - char_len_mask)
        # # [batch_size * max_len, sum_of_cnn_dim]
        # word_emb = tf.reduce_max(outputs, axis=1)

        # highway
        # g = layers.linear(word_emb, cnn_output_dim, activate=tf.nn.relu, name='word_highway_g')
        # t = layers.linear(word_emb, cnn_output_dim, activate=tf.nn.sigmoid, name='word_highway_t')
        # word_emb = t * g + (1-t) * word_emb

        # [batch_size, max_len, sum_of_cnn_dim]
        word_emb = tf.reshape(outputs, [batch_size, max_len, cnn_output_dim])

        # [batch_size, max_len, word_embedding]
        # word_emb = layers.linear(word_emb, self.config.embedding_dim,
        #                          activate=tf.nn.relu, name='word_emb_layers')
        #
        word_emb *= tf.expand_dims(tf.sequence_mask(_lengths, max_len, dtype=tf.float32), axis=-1)

        return word_emb

    def compute_rnn(self, _inputs, _lengths, reuse=True):
        # LSTM Cells
        # lstm cell
        # Create LSTM cell
        def one_lstm_cell():
            if self.config.type == 'lstm':
                c = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, forget_bias=0., reuse=reuse)
            elif self.config.type == 'rnn':
                c = tf.contrib.rnn.BasicRNNCell(self.config.hidden_dim, activation=tf.nn.tanh, reuse=reuse)
            else:
                raise TypeError('undefined rnn type = ' + self.config.type)
            if self.config.dropout > 0:
                c = tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=1. - self.config.dropout)
            return c

        cell_fw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.hidden_layers)])
        cell_bw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.hidden_layers)])

        vocab_size = self.config.vocab_size
        embedding_dim = self.config.embedding_dim

        batch_size = tf.shape(_inputs)[0]

        if self.config.char_nn_type == 'cnn':
            inputs = self.compute_embedding_cnn(_inputs, _lengths, reuse=reuse)  # (batch_size, seq_len, emb_dim)
        elif self.config.char_nn_type == 'lstm':
            inputs = self.compute_embedding_rnn(_inputs, _lengths, reuse=reuse)
        else:
            raise TypeError('undefine the char_nn_type=' + self.config.char_nn_type)

        # dropout
        if self.config.dropout > 0:
            inputs = tf.nn.dropout(inputs, keep_prob=1. - self.config.dropout)

        # lstm
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                          inputs=inputs,
                                                          sequence_length=_lengths,
                                                          initial_state_fw=cell_fw.zero_state(batch_size, tf.float32),
                                                          initial_state_bw=cell_fw.zero_state(batch_size, tf.float32))

        return outputs[0], outputs[1], states



