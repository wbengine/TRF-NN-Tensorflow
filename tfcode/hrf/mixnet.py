import tensorflow as tf

from base import *
from . import net


class Config(net.RNNConfig):
    def __init__(self, vocab_size, output_size):
        super().__init__(vocab_size, output_size)
        self.opt_method = 'adam'
        self.max_update_batch = 100


class Net(object):
    def __init__(self, config, is_training, device='/gpu:0', name='mixnet', reuse=None):
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

            self.net = net.create_net(self.config, self.is_training, self.reuse)
            self.outputs, self.vars = self.net.output(self._inputs, self._lengths, reuse=self.reuse)
            self.var_size = tf.add_n([tf.size(v) for v in self.vars])
            self.phi = self.create_values(self.outputs, self._labels, self._lengths)
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
                                               name=name + '/train_op')

    def print_info(self):
        if self.reuse is None:
            print('[%s.%s] variables in %s' % (__name__, self.__class__.__name__, self.name))
            for v in self.vars:
                print('\t' + v.name, v.shape, v.device)
            # print('[%s.%s] max_update_batch=%d' % (__name__, self.__class__.__name__, self.config.max_update_batch))

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
                         self._grad_outputs: grad_outputs[i: i + self.config.max_update_batch]}
                        )

        # update parameters
        self.train_op.set_lr(session, learning_rate)
        self.train_op.update(session)

    def run_outputs(self, session, inputs, lengths):
        return session.run(self.outputs, {self._inputs: inputs, self._lengths: lengths})

    def run_phi(self, session, inputs, labels, lengths):
        return session.run(self.phi, {self._inputs: inputs, self._lengths: lengths, self._labels: labels})












