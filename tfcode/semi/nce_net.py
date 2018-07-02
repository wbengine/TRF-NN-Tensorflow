import tensorflow as tf
import numpy as np

from base import *
from . import mix_net, alg
from hrf import alg as alg_np


class Config(wb.Config):
    def __init__(self, data):
        self.vocab_size = data.get_vocab_size()
        self.char_vocab_size = data.get_char_size()
        self.output_size = data.get_tag_size()
        self.beg_token = data.get_beg_tokens()[1]
        self.end_token = data.get_end_tokens()[1]
        self.init_weight = 0.1

        self.embedding_size = 200
        self.rnn_hidden_size = 200
        self.rnn_hidden_layers = 1

        self.c2w_type = 'rnn'
        self.chr_embedding_size = 100
        self.c2w_cnn_size = 30
        self.c2w_cnn_width = [1, 2, 3, 4]
        self.c2w_rnn_size = 100
        self.c2w_rnn_layers = 1

        self.dropout = 0

        self.opt_method = 'adam'
        self.max_grad_norm = 10
        self.max_update_batch = 100

        self.noise_factor = 1.0
        self.inter_alpha = 1.0

        self.pi = data.get_pi_true()
        self.init_logz_a = np.log(self.vocab_size)
        self.init_logz_b = -np.log(self.vocab_size)

        self.shared = False

    def __str__(self):
        s = 'noise{:.1f}_'.format(self.noise_factor)
        s += 'blstm_cnn_we{}'.format(self.embedding_size)
        if self.c2w_type is not None:
            s += '_ce{}_c2w{}'.format(self.chr_embedding_size, self.c2w_type)
        if self.dropout > 0:
            s += '_dropout%.1f' % self.dropout
        return s


class Net(object):
    def __init__(self, config, is_training=True, device='/gpu:0', name='mixnet', reuse=None,
                 word_to_chars=None):

        self.config = config
        self.device = device
        self.is_training = is_training
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

            self.crf_phi = None
            self.crf_logz = None
            self.crf_logp = None
            self.trf_phi = None
            self.trf_logz = None
            self.trf_logp = None

            self.create_net(self._inputs, self._labels, self._lengths, word_to_chars)

            # vars
            self.vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope=tf.get_variable_scope().name)
            self.var_size = tf.add_n([tf.size(v) for v in self.vars])
            self.print_info()

            self.saver = tf.train.Saver(self.vars)

            # create loss
            self.crf_loss = - tf.reduce_mean(self.crf_logp)

            self._data_num = tf.placeholder(tf.int32, None, name='data_num')       # integer
            self._noise_logp = tf.placeholder(tf.float32, [None], name='q_logps')  # [batch_size]
            self.nce_loss = self.create_trf_loss(self.trf_logp, self._noise_logp, self._data_num)

            if is_training:

                self.crf_train = layers.TrainOp(self.crf_loss, self.vars, self.config.opt_method,
                                                max_grad_norm=self.config.max_grad_norm,
                                                name=name + '/crf_train_op')

                self.nce_train = layers.TrainOp(self.nce_loss, self.vars, self.config.opt_method,
                                                max_grad_norm=self.config.max_grad_norm,
                                                name=name + '/nce_train_op')

    def print_info(self):
        if self.reuse is None:
            print('[%s.%s] variables in %s' % (__name__, self.__class__.__name__, self.name))
            for v in self.vars:
                print('\t' + v.name, v.shape, v.device)

    def create_word_emb(self, inputs, name='word_embedding'):
        # embedding layers
        word_embedding = tf.get_variable(name,
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

    def create_net(self, inputs, labels, lengths, word_to_chars):

        word_emb = self.create_word_emb(inputs)  # [batch_size, max_len, word_emb]

        if self.config.c2w_type is not None and self.config.chr_embedding_size > 0:
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
                                    dropout=self.config.dropout,
                                    reuse=self.reuse)

        # for CRF
        self.crf_phi, self.crf_logz = self.create_crf_phi(rnn_outputs, labels, lengths)
        self.crf_logp = self.crf_phi - self.crf_logz

        if not self.config.shared:
            word_emb = self.create_word_emb(inputs, name='word_embedding2')  # [batch_size, max_len, word_emb]

            emb = tf.nn.dropout(word_emb, keep_prob=1. - self._dropout)

            # bLSTM
            rnn_outputs, _ = layers.rnn(emb, lengths,
                                        self.config.rnn_hidden_size,
                                        self.config.rnn_hidden_layers,
                                        rnn_type='blstm',
                                        dropout=self.config.dropout,
                                        reuse=self.reuse,
                                        name='rnn2')

        # for TRF
        self.trf_phi, self.trf_logz = self.create_trf_phi(rnn_outputs, emb, lengths, self.crf_logz)
        trf_pi = tf.constant(self.config.pi, dtype=tf.float32, name='pi')
        self.trf_logp = self.trf_phi + tf.log(tf.gather(trf_pi, lengths)) - self.trf_logz



    def create_crf_phi(self, rnn_outputs, labels, lengths):
        outputs = tf.concat(rnn_outputs, axis=-1)

        # linear
        outputs = layers.linear(outputs, self.config.output_size,
                                activate=None, name='final_linear')

        self.emission_output = outputs

        phi, logz = self.create_values(outputs, labels, lengths)
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

    def create_trf_phi(self, rnn_outputs, emb, lengths, crf_logz):
        outputs_fw = rnn_outputs[0]
        outputs_bw = rnn_outputs[1]

        # if emb.shape[-1] != rnn_outputs[0].shape[-1]:
        #     # raise TypeError("[{}.{}] the embedding dim ({}) != rnn dim ({})".format(
        #     #     __name__, self.__class__.__name__, emb.shape[-1], rnn_outputs[0].shape[-1]))
        # add a linear layers
        outputs_fw = layers.linear(outputs_fw, emb.shape[-1].value, activate=tf.nn.relu, name='trf_output_layer_fw')
        outputs_bw = layers.linear(outputs_bw, emb.shape[-1].value, activate=tf.nn.relu, name='trf_output_layer_bw')

        outputs = 0
        if outputs_fw is not None:
            outputs += tf.reduce_sum(outputs_fw[:, 0:-1] * emb[:, 1:], axis=-1)
        if outputs_bw is not None:
            outputs += tf.reduce_sum(outputs_bw[:, 1:] * emb[:, 0:-1], axis=-1)

        outputs *= tf.sequence_mask(lengths - 1, maxlen=tf.shape(emb)[1] - 1, dtype=tf.float32)
        phi = tf.reduce_sum(outputs, axis=-1) + crf_logz  # of shape [batch_size]

        # logz
        self.logz_a = tf.get_variable(name='logz_a', shape=None, dtype=tf.float32,
                                      initializer=np.array(self.config.init_logz_a, 'float32')
                                      )
        self.logz_b = tf.get_variable(name='logz_b', shape=None, dtype=tf.float32,
                                      initializer=np.array(self.config.init_logz_b, 'float32')
                                      )
        logz = self.logz_a * tf.cast(lengths, tf.float32) + self.logz_b

        return phi, logz

    def create_trf_loss(self, mode_logp, noise_logp, data_num):
        noise_logp += np.log(self.config.noise_factor)

        logsum = tf.reduce_logsumexp(tf.stack([mode_logp, noise_logp], axis=0), axis=0)
        logp0 = mode_logp - logsum
        logp1 = noise_logp - logsum

        loss = tf.reduce_mean(logp0[0: data_num]) + \
               self.config.noise_factor * tf.reduce_mean(logp1[data_num:])

        return -loss

    def run_parameter_num(self, session):
        return session.run(self.var_size)

    def run_crf_logp(self, session, inputs, labels, lengths):
        return session.run(self.crf_logp, {self._inputs: inputs, self._lengths: lengths, self._labels: labels})

    def run_trf_logp(self, session, inputs, lengths):
        return session.run(self.trf_logp, {self._inputs: inputs, self._lengths: lengths})

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

    def run_crf_update(self, session, inputs, labels, lengths, learning_rate=1.0):
        self.crf_train.set_lr(session, learning_rate)

        self.crf_train.update(session, {self._inputs: inputs,
                                        self._labels: labels,
                                        self._lengths: lengths,
                                        self._dropout: self.config.dropout})

    def run_trf_update(self, session, inputs, lengths, noise_logps, data_num, learning_rate=1.0):
        self.nce_train.set_lr(session, learning_rate)

        self.nce_train.update(session, {self._inputs: inputs,
                                        self._lengths: lengths,
                                        self._data_num: data_num,
                                        self._noise_logp: noise_logps,
                                        self._dropout: self.config.dropout})

    def save(self, session, fname):
        self.saver.save(session, fname + '.mix.ckpt')

    def restore(self, session, fname):
        self.saver.restore(session, fname + '.mix.ckpt')


class SemiNet(object):
    def __init__(self, config, is_training=True, device='/gpu:0', name='semi-net', word_to_chars=None):
        self.config = config

        self.crf_net = Net(config, is_training=False, device=device, name=name, reuse=None,
                           word_to_chars=word_to_chars)

        self.trf_net = Net(config, is_training=False, device=device, name=name, reuse=True,
                           word_to_chars=word_to_chars)

        self.vars = self.trf_net.vars
        self.var_size = self.trf_net.var_size

        if is_training:

            self.loss = self.trf_net.nce_loss + self.config.inter_alpha * self.crf_net.crf_loss

            self._batch_num = tf.placeholder(tf.int32, shape=None, name='update_batch_size')
            self.grad_clean = []
            self.grad_bufs = []
            for v in self.vars:
                g = tf.get_variable(v.name.split(':')[0] + '_g', shape=v.shape, dtype=tf.float32, trainable=False)
                self.grad_bufs.append(g)
                clean_g = tf.assign(g, tf.zeros_like(g))
                self.grad_clean.append(clean_g)

            grads = tf.gradients(self.loss, self.vars)
            self.grad_update = []
            for g, g_add in zip(self.grad_bufs, grads):
                self.grad_update.append(tf.assign_add(g, g_add / tf.cast(self._batch_num, dtype=tf.float32)))
                # to compute the grad

            # training operation
            self.train_op = layers.TrainOp(self.grad_bufs, self.vars, self.config.opt_method,
                                           max_grad_norm=self.config.max_grad_norm,
                                           name=name + '/train_op')


            # self.train_op = layers.TrainOp(self.loss, self.vars, self.config.opt_method,
            #                                max_grad_norm=self.config.max_grad_norm,
            #                                name=name + '/train_op')

    def run_parameter_num(self, session):
        return session.run(self.var_size)

    def run_crf_logp(self, session, inputs, labels, lengths):
        return self.crf_net.run_crf_logp(session, inputs, labels, lengths)

    def run_trf_logp(self, session, inputs, lengths):
        return self.trf_net.run_trf_logp(session, inputs, lengths)

    def run_opt_labels(self, session, inputs, lengths):
        return self.crf_net.run_opt_labels(session, inputs, lengths)

    def save(self, session, fname):
        self.trf_net.saver.save(session, fname + '.ckpt')

    def restore(self, session, fname):
        self.trf_net.saver.restore(session, fname + '.ckpt')

    def run_update(self, session,
                   crf_inputs, crf_labels, crf_lengths,
                   trf_inputs, trf_lengths, noise_logps, data_num,
                   learning_rate=1.0,
                   max_batch_size=10):

        session.run(self.grad_clean)

        crf_size = crf_inputs.shape[0]
        trf_size = data_num
        batch_num = max(crf_size, trf_size) // max_batch_size

        assert crf_size % batch_num == 0
        assert trf_size % batch_num == 0
        crf_batch_size = crf_size // batch_num
        trf_batch_size = trf_size // batch_num
        noise_batch_size = (trf_inputs.shape[0] - data_num) // batch_num

        def get_trf_data(a, i):
            return np.concatenate([a[i * trf_batch_size: (i+1) * trf_batch_size],
                                   a[data_num + i * noise_batch_size: data_num + (i+1) * noise_batch_size]
                                   ], axis=0)

        for i in range(batch_num):
            session.run(self.grad_update,
                        {self.crf_net._inputs: crf_inputs[i*crf_batch_size: (i+1)*crf_batch_size],
                         self.crf_net._labels: crf_labels[i*crf_batch_size: (i+1)*crf_batch_size],
                         self.crf_net._lengths: crf_lengths[i*crf_batch_size: (i+1)*crf_batch_size],
                         self.trf_net._inputs: get_trf_data(trf_inputs, i),
                         self.trf_net._lengths: get_trf_data(trf_lengths, i),
                         self.trf_net._noise_logp: get_trf_data(noise_logps, i),
                         self.trf_net._data_num: trf_batch_size,
                         self._batch_num: batch_num,
                         self.crf_net._dropout: self.config.dropout,
                         self.trf_net._dropout: 0})

        self.train_op.set_lr(session, learning_rate)
        return self.train_op.update(session)

        # self.train_op.set_lr(session, learning_rate)
        # return self.train_op.update(session, {self.crf_net._inputs: crf_inputs,
        #                                       self.crf_net._labels: crf_labels,
        #                                       self.crf_net._lengths: crf_lengths,
        #                                       self.trf_net._inputs: trf_inputs,
        #                                       self.trf_net._lengths: trf_lengths,
        #                                       self.trf_net._noise_logp: noise_logps,
        #                                       self.trf_net._data_num: data_num,
        #                                       self.crf_net._dropout: self.config.dropout,
        #                                       self.trf_net._dropout: self.config.dropout})


