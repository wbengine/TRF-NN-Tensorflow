import tensorflow as tf
import numpy as np

from base import *
from lm import *
from trf import trfnce
from trf import trfngram


class Config(trfnce.Config):
    def __init__(self, data, lstm_path):
        super().__init__(data)
        if lstm_path is not None:
            lstm_config = lstmlm.load_config(lstm_path)
            self.lstm_path = lstm_path
            self.structure_type = 'rnn'
            self.rnn_type = 'lstm'
            self.rnn_predict = True
            self.embedding_dim = lstm_config.embedding_size
            self.rnn_hidden_size = lstm_config.hidden_size
            self.rnn_hidden_layers = lstm_config.hidden_layers
            self.global_normalized = False
            self.init_zeta = self.get_initial_logz(lstm_config.fixed_logz_for_nce)

        self.feat_type_file = None

    def __str__(self):
        return super().__str__() + '_pretrain'


class NetRNN2(trfnce.NetBase):
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


class TRF(trfnce.TRF):
    def __init__(self, config, data,
                 name='trf', logdir='trf',
                 device='/gpu:0'):
        super().__init__(config, data, name, logdir, device)

        # create saver
        var_dict = {}
        for v in self.train_net.vars:
            name = v.name.split(':')[0]
            tag = name.split('/')[0]
            name = name.replace(tag, 'lstmlm')
            name = name.replace('final_pred', 'BNCELoss')
            var_dict[name] = v
        self.pretrain_saver = tf.train.Saver(var_dict)

    def parser_net(self, net_type):
        if net_type == 'rnn2':
            return NetRNN2
        return super().parser_net(net_type)

    def initialize(self, session):
        super().initialize(session)
        self.load_pretrain_vairables(session, self.config.lstm_path)

    def load_pretrain_vairables(self, session, fname):
        # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        # # List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
        # print_tensors_in_checkpoint_file(file_name=fname, tensor_name='', all_tensors=False)
        print('[TRF] load pretrain variables in ', fname)
        self.pretrain_saver.restore(session, fname)
