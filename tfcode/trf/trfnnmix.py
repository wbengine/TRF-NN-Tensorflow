import tensorflow as tf
from copy import deepcopy
import numpy as np
import time
import json
import os

from . import layers
from . import reader
from . import wblib as wb
from . import trfcnn
from . import trfjsa
from . import word2vec


class Config(trfcnn.Config):
    def __init__(self, data):
        super().__init__(data)
        # for structure
        # rnn structure
        self.rnn_hidden_size = 200
        self.rnn_hidden_layers = 2
        self.rnn_type = 'lstm'
        self.rnn_predict = False
        self.attention = False

    def initial_zeta(self):
        len_num = self.max_len - self.min_len + 1
        logz = np.append(np.zeros(self.min_len),
                         np.linspace(1, len_num, len_num))
        zeta = logz - logz[self.min_len]
        return zeta

    def __str__(self):
        s = 'trf'

        if self.batch_normalize:
            s += '_BN'

        s += '_e{}'.format(self.embedding_dim)

        # cnn structure
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
        s += '_rnn_{}x{}'.format(self.rnn_hidden_size, self.rnn_hidden_layers)
        if self.rnn_predict:
            s += '_pred'

        if self.attention:
            s += '_at'

        return s


class Net(trfcnn.Net):
    def compute_rnn(self, inputs, _lengths, reuse=True):
        # LSTM Cells
        # lstm cell
        # Create LSTM cell
        def one_lstm_cell():
            if self.config.rnn_type == 'lstm':
                c = tf.contrib.rnn.BasicLSTMCell(self.config.rnn_hidden_size, forget_bias=0., reuse=reuse)
            elif self.config.rnn_type == 'rnn':
                c = tf.contrib.rnn.BasicRNNCell(self.config.rnn_hidden_size, activation=tf.nn.tanh, reuse=reuse)
            else:
                raise TypeError('undefined rnn type = ' + self.config.type)
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

        return outputs


class TRF(trfcnn.TRF):
    def __init__(self, config, data, name='trf', logdir='trf', device='/gpu:0',
                 simulater_device=None,
                 q_model=None):
        super().__init__(config, data, name, logdir, device, simulater_device, network=Net, q_model=q_model)