import tensorflow as tf
import numpy as np

from . import reader
from . import layers
from . import word2vec
from . import wblib as wb
from . import net_trf_cnn


class NetTRF(net_trf_cnn.NetTRF):
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
            self.cnn_hidden = 128
            self.cnn_width = 2
            self.cnn_depth = 10  # filter number
            self.cnn_activation = 'relu'  # one of ['relu', 'norm', 'linear']
            self.cnn_use_bias = True
            self.cnn_len_pool = 'max'  # 'max' or 'sum
            self.init_weight = 0.1
            self.load_embedding_path = None

            # for training
            self.opt_method = 'adam'
            self.train_batch_size = 100
            self.sample_batch_size = 100
            self.zeta_gap = 10
            self.max_grad_norm = 10.0

        def __str__(self):
            return 'nntree_h{}x{}_depth{}_{}'.format(self.cnn_hidden, self.cnn_width, self.cnn_depth, self.cnn_activation)

        def initial_zeta(self):
            len_num = self.max_len - self.min_len + 1
            logz = np.append(np.zeros(self.min_len),
                             np.log(self.vocab_size) * np.linspace(1, len_num, len_num))
            zeta = logz - logz[self.min_len]
            return zeta

    def __init__(self, config, reuse=None, is_training=True, name='net_trf', propose_lstm=None):
        super().__init__(config, reuse=reuse, is_training=is_training, name=name)

    def compute_phi(self, _inputs, _lengths, reuse=True):
        """reuse all the variables to compute the phi"""
        vocab_size = self.config.vocab_size
        embedding_dim = self.config.cnn_hidden
        tree_order = self.config.cnn_width
        dbgs = []

        batch_size = tf.shape(_inputs)[0]

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

        ############################################
        # tree parser
        ############################################
        # define the activation
        if self.config.cnn_activation == 'gated':
            use_gated = True
            act = lambda x: x
        else:
            use_gated = False
            act = net_trf_cnn.get_activation(self.config.cnn_activation)
        # variables
        tree_weight_w = tf.get_variable('tree_weight_w',
                                        [self.config.cnn_depth, self.config.cnn_hidden], dtype=tf.float32)
        tree_filter_all = tf.get_variable('tree_filter',
                                          [self.config.cnn_depth,
                                           tree_order, self.config.cnn_hidden, self.config.cnn_hidden],
                                          dtype=tf.float32)
        if use_gated:
            tree_gate_all = tf.get_variable('tree_gate',
                                            [self.config.cnn_depth,
                                             tree_order, self.config.cnn_hidden, self.config.cnn_hidden],
                                            dtype=tf.float32)
        if self.config.cnn_use_bias:
            tree_weight_b = tf.get_variable('tree_weight_b', [self.config.cnn_depth], dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
            tree_filter_bias = tf.get_variable('tree_filter_bias',
                                               [self.config.cnn_hidden], dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
            if use_gated:
                tree_gate_bias = tf.get_variable('tree_gate_bias',
                                                 [self.config.cnn_hidden], dtype=tf.float32,
                                                 initializer=tf.zeros_initializer())

        # condition of loop
        def cond_batch(e, n, depth, phi):
            # do parser if some sentence have not get the root node
            return tf.reduce_any(tf.greater(n, tree_order - 1))

        # parser of loop
        def body_batch(e, n, depth, phi):
            depth_idx = tf.minimum(depth, self.config.cnn_depth-1)
            # e: [batch_size, max_len, hidden_size]
            # y: [batch_size, max_len, hidden_size]
            y = tf.nn.conv1d(
                value=e,
                filters=tree_filter_all[depth_idx],
                stride=1,
                padding='VALID'
            )
            if self.config.cnn_use_bias:
                y += tf.reshape(tree_filter_bias, [1, 1, self.config.cnn_hidden])

            if use_gated:
                # g: gate [batch_size, max_len, hidden_size]
                g = tf.nn.conv1d(
                    value=e,
                    filters=tree_gate_all[depth_idx],
                    stride=1,
                    padding='VALID'
                )
                if self.config.cnn_use_bias:
                    g += tf.reshape(tree_gate_bias, [1, 1, self.config.cnn_hidden])

                # y = act(y)
                y = tf.nn.sigmoid(g) * tf.nn.tanh(y)
            else:
                y = act(y)

            new_n = n - (tree_order - 1)
            old_max_len = tf.shape(e)[1]
            new_max_len = tf.shape(y)[1]

            if self.config.cnn_len_pool == 'sum':
                len_mask = tf.sequence_mask(new_n, new_max_len, dtype=tf.float32)
                y *= tf.expand_dims(len_mask, axis=-1)
                # weights: [batch_size, hidden_size]
                # summation over time
                weights = tf.reduce_sum(y, axis=1)
            elif self.config.cnn_len_pool == 'max':
                len_mask = tf.sequence_mask(new_n, new_max_len, dtype=tf.float32)
                len_mask = tf.expand_dims(len_mask, axis=-1)
                # min value at each dimension, [batch_size, 1, hidden_size]
                min_value = tf.reduce_min(y, axis=1, keep_dims=True)
                # [batch_size, max_len, hidden_size]
                min_value = tf.tile(min_value, [1, new_max_len, 1])

                y = y * len_mask + min_value * (1.0 - len_mask)
                # [batch_size, hidden_size]
                weights = tf.reduce_max(y, axis=1)
            else:
                raise TypeError('undefined pool type = ' + self.config.cnn_len_pool)

            # compute the phi
            weights = weights * tf.expand_dims(tree_weight_w[depth_idx], axis=0)
            if self.config.cnn_use_bias:
                weights += tree_weight_b[depth_idx]

            # if current length is <= 0, then add no weights
            new_phi = tf.where(tf.less_equal(new_n, 0),
                               phi,
                               phi + tf.reduce_sum(weights, axis=-1))

            return y, new_n, depth+1, new_phi

        init_phi = tf.zeros([batch_size], dtype=tf.float32, name='init_phi')

        final_e, final_n, _, final_phi = tf.while_loop(cond=cond_batch,
                                                    body=body_batch,
                                                    loop_vars=[
                                                        emb_output,
                                                        _lengths,
                                                        0,
                                                        init_phi],
                                                    shape_invariants=[
                                                        tf.TensorShape([None, None, self.config.cnn_hidden]),
                                                        tf.TensorShape([None]),  # [batch_size]
                                                        tf.TensorShape([]),
                                                        tf.TensorShape([None]),  # [batch_size]
                                                    ])

        # if self.dbgs is None:
        #     self.dbgs = dbgs

        return final_phi