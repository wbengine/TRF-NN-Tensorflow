import tensorflow as tf

from base import *
from . import alg
from hrf import alg as alg_np


class Config(wb.Config):
    def __init__(self, word_vocab_size, char_vocab_size, output_size, beg_tag_token, end_tag_token):
        self.vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.output_size = output_size
        self.init_weight = 0.1

        self.embedding_size = 100
        self.rnn_hidden_size = 200
        self.rnn_hidden_layers = 1

        self.c2w_type = 'rnn'
        self.chr_embedding_size = 30
        self.c2w_cnn_size = 30
        self.c2w_cnn_width = [1, 2, 3, 4]
        self.c2w_rnn_size = 30
        self.c2w_rnn_layers = 1

        self.dropout = 0

        self.opt_method = 'adam'
        self.max_grad_norm = 10
        self.max_update_batch = 100

        self.beg_token = beg_tag_token
        self.end_token = end_tag_token

    def __str__(self):
        s = 'blstm_cnn_we{}_ce{}_c2w{}'.format(self.embedding_size, self.chr_embedding_size, self.c2w_type)
        if self.dropout > 0:
            s += '_dropout%.1f' % self.dropout
        return s


# only the mix-connection between x and h, without the dependence in h
class Net(object):
    def __init__(self, config, is_training, device='/gpu:0', name='mixnet', reuse=None,
                 word_to_chars=None):

        self.config = config
        self.is_training = is_training
        self.device = device
        self.name = name
        self.reuse = reuse

        default_initializer = tf.random_uniform_initializer(-self.config.init_weight, self.config.init_weight)
        with tf.device(device), tf.variable_scope(self.name, reuse=self.reuse, initializer=default_initializer):
            #############################################
            # inputs: of shape (batch_size, seq_len)
            # labels: of shape (batch_size, seq_len)
            # lengths: of shape (batch_size,)
            #############################################
            self._inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
            self._labels = tf.placeholder(tf.int32, [None, None], name='labels')
            self._lengths = tf.placeholder(tf.int32, [None], name='lengths')
            self._dropout = tf.constant(0, dtype=tf.float32)

            self.phi, self.logz, self.logps = self.create_net(self._inputs, self._labels, self._lengths,
                                                              word_to_chars)

            # vars
            self.vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope=tf.get_variable_scope().name)
            self.var_size = tf.add_n([tf.size(v) for v in self.vars])
            self.print_info()

            self.saver = tf.train.Saver(self.vars)

            if is_training:
                # self.update_vars = self.vars
                #
                # self.grad_clean = []
                # self.grad_bufs = []
                # for v in self.update_vars:
                #     g = tf.get_variable(v.name.split(':')[0] + '_g', shape=v.shape, dtype=tf.float32, trainable=False)
                #     self.grad_bufs.append(g)
                #     clean_g = tf.assign(g, tf.zeros_like(g))
                #     self.grad_clean.append(clean_g)
                #
                # self._data_num = tf.placeholder(dtype=tf.int32, shape=None, name='data_num')
                #
                # data_phi = self.phi[0: self._data_num]
                # samp_phi = self.phi[self._data_num:]

                # grads = tf.gradients(self.outputs, self.update_vars, self._grad_outputs)
                # self.grad_update = []
                # for g, g_add in zip(self.grad_bufs, grads):
                #     self.grad_update.append(tf.assign_sub(g, g_add))  # to compute the -grad

                # training operation
                self.loss = -tf.reduce_mean(self.logps)
                self.train_op = layers.TrainOp(self.loss, self.vars, self.config.opt_method,
                                               max_grad_norm=self.config.max_grad_norm,
                                               name=name + '/train_op')

    def print_info(self):
        if self.reuse is None:
            print('[%s.%s] variables in %s' % (__name__, self.__class__.__name__, self.name))
            for v in self.vars:
                print('\t' + v.name, v.shape, v.device)

    def create_net(self, inputs, labels, lengths, word_to_chars):

        # linear
        outputs = self.create_emission(inputs, lengths, word_to_chars)
        self.emission_output = outputs

        # phi
        phi, logz = self.create_values(outputs, labels, lengths)

        # logps
        logps = phi - logz

        return phi, logz, logps

    def create_emission(self, inputs, lengths, word_to_chars):
        word_emb = self.create_word_emb(inputs)  # [batch_size, max_len, word_emb]

        if self.config.chr_embedding_size > 0:
            char_emb = self.create_char_emb(inputs, word_to_chars, self.reuse)
            char_emb = tf.nn.dropout(char_emb, keep_prob=1. - self._dropout)
            emb = tf.concat([word_emb, char_emb], axis=-1)
        else:
            emb = word_emb

        emb = tf.nn.dropout(emb, keep_prob=1. - self._dropout)

        # bLSTM
        rnn_outputs, _ = layers.rnn(emb, lengths,
                                    self.config.rnn_hidden_size,
                                    self.config.rnn_hidden_layers,
                                    rnn_type='blstm',
                                    dropout=self.config.dropout if self.is_training else None,
                                    reuse=self.reuse)
        rnn_outputs = tf.concat(rnn_outputs, axis=-1)
        rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob=1. - self._dropout)

        # linear
        outputs = layers.linear(rnn_outputs, self.config.output_size,
                                activate=None, name='final_linear')

        return outputs

    def create_word_emb(self, inputs):
        # embedding layers
        word_embedding = tf.get_variable('word_embedding',
                                         [self.config.vocab_size, self.config.embedding_size], dtype=tf.float32)
        emb = tf.nn.embedding_lookup(word_embedding, inputs)  # (batch_size, seq_len, emb_dim)
        return emb

    def create_char_emb(self, inputs, word_to_chars, reuse=None):

        if self.config.c2w_type == 'cnn':
            return layers.char_emb_cnn(inputs, char_size=self.config.char_vocab_size,
                                       embedding_size=self.config.chr_embedding_size,
                                       cnn_kernel_size=self.config.c2w_cnn_size,
                                       cnn_kernel_width=self.config.c2w_cnn_width,
                                       word_to_chars=word_to_chars,
                                       reuse=reuse)
        else:
            return layers.char_emb_rnn(inputs, char_size=self.config.char_vocab_size,
                                       embedding_size=self.config.chr_embedding_size,
                                       rnn_hidden_size=self.config.c2w_rnn_size,
                                       rnn_hidden_layers=self.config.c2w_rnn_layers,
                                       word_to_chars=word_to_chars,
                                       reuse=reuse)

    def create_values2(self, outputs, labels, lengths):
        batch_size = tf.shape(outputs)[0]
        max_length = tf.shape(outputs)[1]
        output_dim = self.config.rnn_hidden_size * 2

        # weight between adjacent h_i, h_i+1
        self.labels_vecs = tf.get_variable('labels_vecs',
                                           [self.config.output_size,
                                            self.config.output_size,
                                            output_dim], dtype=tf.float32)
        self.labels_bias = tf.get_variable('labels_bias',
                                           [self.config.output_size,
                                            self.config.output_size], dtype=tf.float32)

        idx1 = tf.reshape(labels[:, 0: -1], [-1])
        idx2 = tf.reshape(labels[:, 1:], [-1])
        idx = tf.stack([idx1, idx2], axis=1)
        cur_vecs = tf.reshape(tf.gather_nd(self.labels_vecs, idx), [batch_size, -1, output_dim])
        # [batch_size, len-1, rnn_hidden_size]
        cur_bias = tf.reshape(tf.gather_nd(self.labels_bias, idx), [batch_size, -1])
        # [batch_size, len-1]
        values = tf.reduce_sum(cur_vecs * outputs[:, 1:], axis=-1) + cur_bias
        len_mask = tf.sequence_mask(lengths - 1, maxlen=max_length - 1, dtype=tf.float32)  # length-1
        phi = tf.reduce_sum(values * len_mask, axis=-1)  # [batch_size]

        # logz
        # [batch_size, len, tag_size, tag_size, dim]
        trans_vecs = tf.reshape(self.labels_vecs,
                                [1, 1,
                                 self.config.output_size, self.config.output_size, output_dim])
        trans_bias = tf.reshape(self.labels_bias,
                                [1, 1, self.config.output_size, self.config.output_size])

        trans_out = tf.reshape(outputs[:, 1:],
                               [batch_size, max_length-1, 1, 1, -1])

        self.trans = tf.reduce_sum(trans_vecs * trans_out, axis=-1) + trans_bias

        alphas = alg.forward_tf2(self.trans, lengths,
                                 beg_ids=[self.config.beg_token],
                                 end_ids=[self.config.end_token])
        logz = alg.get_logsum(alphas, lengths)

        return phi, logz

    def create_values(self, outputs, labels, lengths):
        # weight between x and h
        batch_size = tf.shape(labels)[0]
        max_length = tf.shape(labels)[1]

        labels_flat = tf.reshape(labels, [-1])
        outputs_flat = tf.reshape(outputs, [-1, tf.shape(outputs)[-1]])
        idx = tf.stack([tf.range(tf.shape(labels_flat)[0]), labels_flat], axis=1)
        loss = tf.gather_nd(outputs_flat, idx)
        loss = tf.reshape(loss, [batch_size, -1])

        len_mask = tf.sequence_mask(lengths, maxlen=max_length, dtype=tf.float32)
        loss = tf.reduce_sum(loss * len_mask, axis=-1)  # [batch_size]

        # weight between adjacent h_i, h_i+1
        self.edge_matrix = tf.get_variable('edge_mat', [self.config.output_size, self.config.output_size],
                                           dtype=tf.float32)
        edge_idx1 = tf.reshape(labels[:, 0:-1], [-1])
        edge_idx2 = tf.reshape(labels[:, 1:], [-1])
        idx = tf.stack([edge_idx1, edge_idx2], axis=1)
        values = tf.gather_nd(self.edge_matrix, idx)
        values = tf.reshape(values, [batch_size, -1])

        len_mask = tf.sequence_mask(lengths - 1, maxlen=max_length - 1, dtype=tf.float32)  # length-1
        loss_bigram = tf.reduce_sum(values * len_mask, axis=-1)  # [batch_size]

        phi = loss + loss_bigram

        # logz
        alphas = alg.forward_tf(self.edge_matrix, self.emission_output, lengths,
                                beg_ids=[self.config.beg_token],
                                end_ids=[self.config.end_token])
        logz = alg.get_logsum(alphas, lengths)

        return phi, logz

    def run_parameter_num(self, session):
        return session.run(self.var_size)

    def run_update(self, session, inputs, labels, lengths, learning_rate=1.0):

        # update parameters
        self.train_op.set_lr(session, learning_rate)

        self.train_op.update(session, {self._inputs: inputs,
                                       self._labels: labels,
                                       self._lengths: lengths,
                                       self._dropout: self.config.dropout})

    def run_outputs(self, session, inputs, lengths):
        return session.run(self.emission_output, {self._inputs: inputs, self._lengths: lengths})

    def run_phi(self, session, inputs, labels, lengths):
        return session.run(self.phi, {self._inputs: inputs, self._lengths: lengths, self._labels: labels})

    def run_logz(self, session, inputs, lengths):
        return session.run(self.logz, {self._inputs: inputs, self._lengths: lengths})

    def run_logp(self, session, inputs, labels, lengths):
        return session.run(self.logps, {self._inputs: inputs, self._lengths: lengths, self._labels: labels})

    def run_opt_labels(self, session, inputs, lengths):
        trans_mat = session.run(self.edge_matrix)
        emiss_mats = session.run(self.emission_output, {self._inputs: inputs, self._lengths: lengths})

        label_list = []
        for emat, n in zip(emiss_mats, lengths):
            fb = alg_np.ForwardBackward(trans_mat, trans_mat, emat[0:n],
                                        [self.config.beg_token],
                                        [self.config.end_token])
            label, _ = fb.decode()
            label_list.append(label)

        return label_list

        # trans_mat = session.run(self.trans, {self._inputs: inputs, self._lengths: lengths})
        # label_list = []
        # for trans, n in zip(trans_mat, lengths):
        #     label, _ = alg_np.decode(trans[0:n-1], [self.config.beg_token], [self.config.end_token])
        #     label_list.append(label)
        #
        # return label_list

    def save(self, session, fname):
        self.saver.save(session, fname + '.mix.ckpt')

    def restore(self, session, fname):
        self.saver.restore(session, fname + '.mix.ckpt')
