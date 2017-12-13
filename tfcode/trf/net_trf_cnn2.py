import tensorflow as tf
import numpy as np

from . import reader
from . import net_trf_rnn
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


class NetTRF(net_trf_rnn.NetTRF):
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
            self.only_train_lambda = False

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

            if self.only_train_lambda:
                s += '_part'
            return s

        def initial_zeta(self):
            len_num = self.max_len - self.min_len + 1
            logz = np.append(np.zeros(self.min_len),
                             np.log(self.vocab_size) * np.linspace(1, len_num, len_num))
            zeta = logz - logz[self.min_len]
            return zeta

    def __init__(self, config, reuse=None, is_training=True, name='net_trf', propose_lstm=None):
        super().__init__(config, reuse=reuse, is_training=is_training, name=name, propose_lstm=propose_lstm)

    def compute_cnn(self, _inputs, _lengths, reuse=True):
        # compute the output of cnn
        """reuse all the variables to compute the phi"""
        vocab_size = self.config.vocab_size
        embedding_dim = self.config.embedding_dim
        batch_size = tf.shape(_inputs)[0]
        max_len = tf.shape(_inputs)[1]
        # the length mask, the position < len is 1., otherwise is 0.
        len_mask, expand_len_mask = layers.generate_length_mask(_lengths, batch_size, max_len)

        # define the activation
        def create_cnn(inputs, dim, width, name):
            # perform cnn
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
            return outputs * expand_len_mask

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

        # dropout
        if self.is_training and self.config.dropout > 0:
            emb_output = tf.nn.dropout(emb_output, keep_prob=1. - self.config.dropout)

        # cnn back
        # compute cnn with different filter width
        inputs = emb_output * expand_len_mask
        conv_list = []
        for (filter_width, out_dim) in self.config.cnn_banks:
            # [batch_size, max_len, out_dim]
            conv_output = create_cnn(inputs, out_dim, filter_width, name='cnn0_%d' % filter_width)
            conv_list.append(conv_output)

        if conv_list:
            inputs = tf.concat(conv_list, axis=-1)

        # cnn stack
        skip_connections = []
        for i, (filter_width, out_dim) in enumerate(self.config.cnn_stacks):
            conv_output = create_cnn(inputs, out_dim, filter_width, 'cnn%d' % (i + 1))

            if self.config.cnn_skip_connection:
                skip_connections.append(conv_output)

            inputs = conv_output

        # skip connections
        if skip_connections:
            inputs = tf.concat(skip_connections, axis=-1)

        # return [batch_size, max_len, dim]
        return inputs

    def compute_phi(self, _inputs, _lengths, reuse=True):
        """reuse all the variables to compute the phi"""

        # [batch_size, max_len, dim]
        outputs = self.compute_cnn(_inputs, _lengths, reuse)

        # [batch_size, max_len, 1]
        outputs = layers.linear(outputs, 1, name='final_layers')

        # [batch_size, max_len]
        outputs = tf.reshape(outputs, [tf.shape(_inputs)[0], -1])

        # [batch_size]
        outputs = outputs * tf.sequence_mask(_lengths, maxlen=tf.shape(_inputs)[1], dtype=tf.float32)
        outputs = tf.reduce_sum(outputs, axis=-1)  # of shape [batch_size]

        return outputs

    def get_pre_train(self, _inputs, _lengths, reuse=True):
        with tf.variable_scope(self.name, reuse=True, initializer=self.default_initializer):
            outputs = self.compute_cnn(_inputs, _lengths, True)

        with tf.variable_scope(self.name, reuse=reuse, initializer=self.default_initializer):
            if self.config.vocab_size <= 50000:
                softmax_fw = layers.Softmax(inputs=outputs[:, 0: -1],
                                            labels=_inputs[:, 1:],
                                            vocab_size=self.config.vocab_size,
                                            name='Softmax_fw')
                softmax_bw = layers.Softmax(inputs=outputs[:, 1:],
                                            labels=_inputs[:, 0:-1],
                                            vocab_size=self.config.vocab_size,
                                            name='Softmax_bw')
            else:
                # using the shortlist softmax
                softmax = layers.ShortlistSoftmax(inputs=outputs[:, 0: -1],
                                                  labels=_inputs[:, 1:],
                                                  shortlist=[10000, self.config.vocab_size],
                                                  name='Softmax_Shortlist')

            # loss is the summation of all inputs
            max_len = tf.shape(_inputs)[1]
            loss = (softmax_fw.loss + softmax_bw.loss) * tf.sequence_mask(_lengths-1, maxlen=max_len-1, dtype=tf.float32)
            loss = tf.reduce_sum(loss)

            # with tf.variable_scope(self.name, reuse=reuse, initializer=self.default_initializer):
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            train = layers.TrainOp(loss, tvars, 'sgd', max_grad_norm=10, initial_lr=1.0, name='pretrain_op')

            return train

    def get_vars(self):
        if self.config.only_train_lambda:
            all_vars = super().get_vars()
            tvars = []
            for v in all_vars:
                if v.name.find('final_layers') >= 0:
                    tvars.append(v)
            return tvars
        else:
            return super().get_vars()



