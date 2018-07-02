import tensorflow as tf

from base import *
# from . import net


class Config(wb.Config):
    def __init__(self, vocab_size, output_size):
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.init_weight = 0.1

        self.embedding_size = 200
        self.rnn_hidden_size = 200
        self.rnn_hidden_layers = 1

        self.chr_embedding_size = 30
        self.cnn_size = 30
        self.cnn_width = 3

        self.dropout = 0

        self.opt_method = 'adam'
        self.max_grad_norm = 10
        self.max_update_batch = 100

    def __str__(self):
        return 'blstm_cnn_we{}_ce{}'.format(self.embedding_size, self.chr_embedding_size)


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
            self._dropout = tf.constant(0.0, dtype=tf.float32)

            # self.net = net.create_net(self.config, self.is_training, self.reuse)
            # self.outputs, self.vars = self.net.output(self._inputs, self._lengths, reuse=self.reuse)
            self.word_emb = self.create_word_emb(self._inputs)  # [batch_size, max_len, word_emb]
            self.char_emb = self.create_char_emb(self._inputs, word_to_chars, self.reuse)
            self.emb = tf.concat([self.word_emb, self.char_emb], axis=-1)

            # bLSTM
            rnn_outputs, _ = layers.rnn(tf.nn.dropout(self.emb, keep_prob=1. - self._dropout),
                                        self._lengths,
                                        self.config.rnn_hidden_size,
                                        self.config.rnn_hidden_layers,
                                        rnn_type='blstm',
                                        dropout=None,
                                        reuse=self.reuse)
            rnn_outputs = tf.concat(rnn_outputs, axis=-1)
            rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob=1. - self._dropout)

            # linear
            self.outputs = layers.linear(rnn_outputs, self.config.output_size,
                                         activate=None, name='final_linear')

            # phi
            self.phi = self.create_values(self.outputs, self._labels, self._lengths)

            # vars
            self.vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope=tf.get_variable_scope().name)
            self.var_size = tf.add_n([tf.size(v) for v in self.vars])
            self.print_info()

            if is_training:
                self.update_vars = self.vars

                self.grad_clean = []
                self.grad_bufs = []
                for v in self.update_vars:
                    g = tf.get_variable(v.name.split(':')[0] + '_g', shape=v.shape, dtype=tf.float32, trainable=False)
                    self.grad_bufs.append(g)
                    clean_g = tf.assign(g, tf.zeros_like(g))
                    self.grad_clean.append(clean_g)

                # _inputs: all the input sequences, data_seqs + sample_seqs
                # _lengths: lengths of the input sequences
                # _grad_outputs: dphi/dy for each sequence. [batch_size, max_length, self.config.output_size]
                #           for data seqs, each is a one-hot / train_batch_size
                #           for sample seqs, - 1.0/sample_batch_size * pi_true[l]/pi_0[l] * marginal_probs
                self._grad_outputs = tf.placeholder(tf.float32, [None, None, self.config.output_size],
                                                    name='grad_outputs')

                grads = tf.gradients(self.outputs, self.update_vars, self._grad_outputs)
                self.grad_update = []
                for g, g_add in zip(self.grad_bufs, grads):
                    self.grad_update.append(tf.assign_sub(g, g_add))  # to compute the -grad

                # training operation
                self.train_op = layers.TrainOp(self.grad_bufs, self.update_vars, self.config.opt_method,
                                               max_grad_norm=self.config.max_grad_norm,
                                               name=name + '/train_op')

    def print_info(self):
        if self.reuse is None:
            print('[%s.%s] variables in %s' % (__name__, self.__class__.__name__, self.name))
            for v in self.vars:
                print('\t' + v.name, v.shape, v.device)
                # print('[%s.%s] max_update_batch=%d' % (__name__, self.__class__.__name__, self.config.max_update_batch))

    def create_word_emb(self, inputs):
        # embedding layers
        word_embedding = tf.get_variable('word_embedding',
                                         [self.config.vocab_size, self.config.embedding_size], dtype=tf.float32)
        emb = tf.nn.embedding_lookup(word_embedding, inputs)  # (batch_size, seq_len, emb_dim)
        return emb

    def create_char_emb(self, inputs, word_to_chars, reuse=None):
        """
        :param inputs: placeholder
        :param word_to_chars: a list of list
        :param reuse: reuse in variable_scope
        :return: char embedding
        """
        batch_size = tf.shape(inputs)[0]
        max_length = tf.shape(inputs)[1]

        char_arys, char_lens = reader.produce_data_to_array(word_to_chars)
        char_max_len = char_arys.shape[1]
        self.char_arys = tf.constant(char_arys, name='char_arys')
        self.char_lens = tf.constant(char_lens, name='char_lens')

        char_inputs = tf.gather(self.char_arys, tf.reshape(inputs, [-1]))  # [word_num, char_max_num]
        char_lens = tf.gather(self.char_lens, tf.reshape(inputs, [-1]))  # [word_num]
        char_mask = tf.sequence_mask(char_lens, maxlen=char_max_len, dtype=tf.float32)  # [word_num, char_max_len]
        char_mask_ext = tf.expand_dims(char_mask, axis=-1)  # [word_num, char_max_len, 1]

        # embedding
        char_embedding = tf.get_variable('char_embedding',
                                         [self.config.vocab_size, self.config.embedding_size], dtype=tf.float32)
        emb = tf.nn.embedding_lookup(char_embedding, char_inputs)  # (word_num, char_max_len, char_emb_dim)
        emb *= char_mask_ext

        # CNN
        conv = tf.layers.conv1d(
            inputs=emb,  # set the values at positon >= length to zeros
            filters=self.config.cnn_size,
            kernel_size=self.config.cnn_width,
            padding='same',
            activation=tf.nn.relu,
            reuse=reuse,
            name='cnn0'
        )
        conv *= char_mask_ext  # (word_num, char_max_len, dim)

        # max-pooling
        outputs = tf.reduce_max(conv, axis=1)  # (word_num, dim)
        outputs = tf.reshape(outputs, [batch_size, max_length, self.config.cnn_size])  # (batch_size, max_len, dim)

        return outputs

    def create_values(self, outputs, labels, lengths):
        batch_size = tf.shape(labels)[0]
        max_length = tf.shape(labels)[1]

        labels = tf.reshape(labels, [-1])
        outputs = tf.reshape(outputs, [-1, tf.shape(outputs)[-1]])
        idx = tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1)
        loss = tf.gather_nd(outputs, idx)
        loss = tf.reshape(loss, [batch_size, -1])

        # mask
        len_mask = tf.sequence_mask(lengths, maxlen=max_length, dtype=tf.float32)
        loss = tf.reduce_sum(loss * len_mask, axis=-1)  # [batch_size]

        return loss

    def run_parameter_num(self, session):
        return session.run(self.var_size)

    def run_update(self, session, inputs, lengths, grad_outputs, learning_rate=1.0):

        # clean gradient
        session.run(self.grad_clean)

        # compute the gradient
        for i in range(0, len(inputs), self.config.max_update_batch):
            session.run(self.grad_update,
                        {self._inputs: inputs[i: i + self.config.max_update_batch],
                         self._lengths: lengths[i: i + self.config.max_update_batch],
                         self._grad_outputs: grad_outputs[i: i + self.config.max_update_batch],
                         self._dropout: self.config.dropout}
                        )

        # update parameters
        self.train_op.set_lr(session, learning_rate)
        self.train_op.update(session)

    def run_outputs(self, session, inputs, lengths):
        return session.run(self.outputs, {self._inputs: inputs, self._lengths: lengths})

    def run_phi(self, session, inputs, labels, lengths):
        return session.run(self.phi, {self._inputs: inputs, self._lengths: lengths, self._labels: labels})

# # add the bigram in h
# class Net2(Net):
#     def __init__(self, config, is_training, device='/gpu:0', name='mixnet', reuse=None):
#         super().__init__(config, is_training, device, name, reuse)
#
#     def create_values(self, outputs, labels, lengths):
#
#         # weight between x and h
#         loss_mix = super().create_values(outputs, labels, lengths)
#
#         # weight between adjacent h_i, h_i+1
#         batch_size = tf.shape(labels)[0]
#         max_length = tf.shape(labels)[1]
#
#         self.edge_matrix = tf.get_variable('edge_mat', shape=[self.config.output_size, self.config.output_size],
#                                            dtype=tf.float32)
#         self.vars.append(self.edge_matrix)
#
#         edge_idx1 = tf.reshape(labels[:, 0:-1], [-1])
#         edge_idx2 = tf.reshape(labels[:, 1:], [-1])
#         idx = tf.stack([edge_idx1, edge_idx2], axis=1)
#         values = tf.gather_nd(self.edge_matrix, idx)
#         values = tf.reshape(values, [batch_size, -1])
#
#         len_mask = tf.sequence_mask(lengths, maxlen=max_length-1, dtype=tf.float32)  # length-1
#         loss = tf.reduce_sum(values * len_mask, axis=-1)  # [batch_size]
#
#         return loss + loss_mix
