import tensorflow as tf

from base import *


def create_net(net_config, is_training, reuse):
    if isinstance(net_config, RNNConfig):
        return NetRNN(net_config, is_training, reuse)
    else:
        raise TypeError('Unknown net_config tpye = {}'.format(type(net_config)))


class BaseConfig(wb.Config):
    def __init__(self, vocab_size, output_size):
        self.vocab_size = vocab_size
        self.output_size = output_size

        self.embedding_size = 128
        self.init_weight = 0.1
        self.dropout = 0


class RNNConfig(BaseConfig):
    def __init__(self, vocab_size, output_size):
        super().__init__(vocab_size, output_size)

        self.embedding_size = 200
        self.hidden_size = 200
        self.hidden_layers = 1
        self.rnn_type = 'blstm'

    def __str__(self):
        return self.rnn_type + '_e{}_h{}x{}'.format(self.embedding_size, self.hidden_size, self.hidden_layers)


class BaseNet(object):
    def __init__(self, net_config, is_training, reuse=None):
        self.config = net_config
        self.is_training = is_training
        self.reuse = reuse

    def output(self, _inputs, _lengths, reuse=None):
        """return the outputs and the variables"""
        pass


class NetRNN(BaseNet):
    def __init__(self, net_config, is_training, reuse=None):
        super().__init__(net_config, is_training, reuse)

    def compute_rnn(self, _inputs, _lengths, reuse=True):

        def one_lstm_cell():
            if self.config.rnn_type.find('lstm') != -1:
                c = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=0., reuse=reuse)
            elif self.config.rnn_type.find('rnn') != -1:
                c = tf.contrib.rnn.BasicRNNCell(self.config.hidden_size, activation=tf.nn.tanh, reuse=reuse)
            else:
                raise TypeError('undefined rnn type = ' + self.config.rnn_type)
            if self.is_training and self.config.dropout > 0:
                c = tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=1. - self.config.dropout)
            return c

        batch_size = tf.shape(_inputs)[0]

        # embedding layers
        word_embedding = tf.get_variable('word_embedding',
                                         [self.config.vocab_size, self.config.embedding_size], dtype=tf.float32)
        emb = tf.nn.embedding_lookup(word_embedding, _inputs)  # (batch_size, seq_len, emb_dim)
        inputs = emb

        # dropout
        if self.is_training and self.config.dropout > 0:
            inputs = tf.nn.dropout(inputs, keep_prob=1. - self.config.dropout)

        # recurrent structure
        if self.config.rnn_type[0].lower() == 'b':
            cell_fw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.hidden_layers)])
            cell_bw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.hidden_layers)])
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                              inputs=inputs,
                                                              sequence_length=_lengths,
                                                              dtype=tf.float32)
            outputs_fw = outputs[0]
            outputs_bw = outputs[1]

        else:
            cell_fw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.hidden_layers)])
            outputs, states = tf.nn.dynamic_rnn(cell_fw,
                                                inputs=inputs,
                                                sequence_length=_lengths - 1)
            outputs_fw = outputs
            outputs_bw = None

        return outputs_fw, outputs_bw, states, emb

    def output(self, _inputs, _lengths, reuse=None):

        outputs_fw, outputs_bw, _, emb = self.compute_rnn(_inputs, _lengths, reuse)
        outputs = tf.concat([outputs_fw, outputs_bw], axis=2)

        # [batch_size, max_len, output_size]
        outputs = layers.linear(outputs, self.config.output_size, activate=None, name='final_linear')
        #
        # labels = tf.reshape(_labels, [-1])
        # idx = tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1)
        # loss = tf.gather_nd(outputs, idx)
        # loss = tf.reshape(loss, [batch_size, -1])
        #
        # # mask
        # len_mask = tf.sequence_mask(_lengths, maxlen=max_length, dtype=tf.float32)
        # outputs = layers.linear(inputs, 1, name='final_linear')  # [batch_size, max_len, 1]
        # outputs = tf.reshape(outputs, [batch_size, -1])  # [batch_size, max_len]
        # outputs = outputs * len_mask
        # outputs = tf.reduce_sum(outputs, axis=-1)  # of shape [batch_size]

        return outputs, tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)














