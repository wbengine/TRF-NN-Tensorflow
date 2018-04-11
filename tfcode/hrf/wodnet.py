import tensorflow as tf

from base import *
from trf.common import net


class Config(net.Config):
    def __init__(self, vocab_size):
        super().__init__(vocab_size)
        self.max_update_batch = 100  # the max-batch used in update
        self.max_grad_norm = 10
        self.opt_method = 'sgd'

        self.only_train_weight = False  # if true, then rm the embedding from the variable list


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
                print('[%s.%s] max_update_batch=%d' % (__name__, self.__class__.__name__, self.config.max_update_batch))

            ###################################################
            # for SA training
            ###################################################
            if is_training:
                # a variables used to save the gradient
                if self.config.only_train_weight:
                    self.update_vars = self.vars[-2:]
                else:
                    self.update_vars = self.vars

                # print('[%s.%s] update variables in %s' % (__name__, self.__class__.__name__, name))
                # for v in self.update_vars:
                #     print('\t' + v.name, v.shape, v.device)

                self.grad_clean = []
                self.grad_bufs = []
                for v in self.update_vars:
                    g = tf.get_variable(v.name.split(':')[0] + '_g', shape=v.shape, dtype=tf.float32, trainable=False)
                    self.grad_bufs.append(g)
                    clean_g = tf.assign(g, tf.zeros_like(g))
                    self.grad_clean.append(clean_g)

                # _inputs: all the input sequences, data_seqs + sample_seqs
                # _lengths: lengths of the input sequences
                # _scalers: the scalers of the input sequences
                #           for data seqs, scaler = 1.0/data_batch_size
                #           for sample seqs: scaler = - 1.0/sample_batch_size * pi_true[l]/pi_0[l]
                self._scalers = tf.placeholder(tf.float32, [batch_size], name='input_scalers')

                # compute d\phi / d\theta * scaler and add to the grad_vars
                grads = tf.gradients(tf.reduce_sum(self.phi * self._scalers), self.update_vars)
                self.grad_update = []
                for g, g_add in zip(self.grad_bufs, grads):
                    self.grad_update.append(tf.assign_sub(g, g_add))  # to compute the -grad

                ###################################
                # training
                ###################################
                self.trainop = layers.TrainOp(self.grad_bufs, self.update_vars,
                                              optimize_method=self.config.opt_method,
                                              max_grad_norm=self.config.max_grad_norm,
                                              name=name + '/train_op')

    def get_param_num(self):
        return tf.get_default_session().run(self.var_size)

    def get_phi(self, seq_list):
        inputs, lengths = reader.produce_data_to_array(seq.get_x(seq_list))
        return tf.get_default_session().run(self.phi,
                                            {self._inputs: inputs,
                                             self._lengths: lengths})

    def update(self, data_list, data_scalers, sample_list, sample_scalers, learning_rate):

        total_seqs = seq.get_x(data_list) + seq.get_x(sample_list)
        total_inputs, total_lengths = reader.produce_data_to_array(total_seqs)
        total_scalers = np.concatenate([data_scalers, -np.array(sample_scalers)])

        # clean gradient
        tf.get_default_session().run(self.grad_clean)

        # compute the gradient
        for i in range(0, len(total_seqs), self.config.max_update_batch):
            tf.get_default_session().run(self.grad_update,
                                         {self._inputs: total_inputs[i: i+self.config.max_update_batch],
                                          self._lengths: total_lengths[i: i+self.config.max_update_batch],
                                          self._scalers: total_scalers[i: i+self.config.max_update_batch]}
                                         )

        # update parameters
        self.trainop.set_lr(tf.get_default_session(), learning_rate=learning_rate)
        self.trainop.update(tf.get_default_session(), None)
