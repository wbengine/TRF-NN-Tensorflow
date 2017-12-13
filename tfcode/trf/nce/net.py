import tensorflow as tf

from base import *
from trf.common import net


class Config(net.Config):
    def __init__(self, data):
        super().__init__(data.get_vocab_size())
        self.max_grad_norm = 10
        self.opt_method = 'sgd'


class NetBase(object):
    def __init__(self, config, is_training, device='/gpu:0', name='net', reuse=None):
        self.is_training = is_training
        self.config = config

        # if training, the batch_size is fixed, make the training more effective
        # batch_size = self.config.batch_size * (1 + self.config.noise_factor) if is_training else None
        batch_size = None
        default_initializer = tf.random_uniform_initializer(-config.init_weight, config.init_weight)
        with tf.device(device), tf.variable_scope(name, reuse=reuse, initializer=default_initializer):
            #############################################
            # inputs: of shape (batch_size, seq_len)
            # lengths: of shape (batch_size,)
            #############################################
            self._inputs = tf.placeholder(tf.int32, [batch_size, None], name='inputs')
            self._lengths = tf.placeholder(tf.int32, [batch_size], name='lengths')
            self._q_logps = tf.placeholder(tf.float32, [batch_size], name='q_logps')
            #############################################################
            # compute the energy function phi
            #############################################################
            self.net = net.create_net(config, is_training, reuse)
            self.phi, self.vars = self.net.output(self._inputs, self._lengths, reuse=reuse)
            self.var_size = tf.add_n([tf.size(v) for v in self.vars])
            # print the variables
            if reuse is None:
                print('[%s.%s] variables in %s' % (__name__, self.__class__.__name__, name))
                for v in self.vars:
                    print('\t' + v.name, v.shape, v.device)

            ###################################################
            # for NCE training
            ###################################################
            if is_training:
                #############################################################
                # compute the loss
                #############################################################
                # the noise log-probs for all the input sequences, of shape [batch_size]
                self._cluster_weights = tf.placeholder(tf.float32, [batch_size], name='input_cluster_weights')
                self.grads = tf.gradients(self.phi, self.vars, -self._cluster_weights)

                ###################################
                # training
                ###################################
                self.trainop = layers.TrainOp(self.grads, self.vars,
                                              optimize_method=self.config.opt_method,
                                              max_grad_norm=self.config.max_grad_norm,
                                              name=name + '/train_op')

    def get_param_num(self):
        return tf.get_default_session().run(self.var_size)

    def get_phi(self, seq_list):
        inputs, lengths = reader.produce_data_to_array(seq_list)
        return tf.get_default_session().run(self.phi,
                                            {self._inputs: inputs,
                                             self._lengths: lengths})

    def update(self, seq_list, cluster_weights, cluster_m=None, learning_rate=1.0):
        inputs, lengths = reader.produce_data_to_array(seq_list)
        # set learning rate
        self.trainop.set_lr(tf.get_default_session(), learning_rate)
        # update parameters
        self.trainop.update(tf.get_default_session(), {self._inputs: inputs,
                                                       self._lengths: lengths,
                                                       self._cluster_weights: cluster_weights})