import tensorflow as tf

from base import *
from trf.common import net


class Config(net.Config):
    def __init__(self, data):
        super().__init__(data.get_vocab_size())
        self.max_grad_norm = 10
        self.opt_method = 'sgd'
        self.l2_reg = 0

        self.pi = data.get_pi_true()

        self.init_logz_a = np.log(self.vocab_size)
        self.init_logz_b = -np.log(self.vocab_size)

        self.fdiv_type = 'GAN'

    def __str__(self):
        s = self.fdiv_type + '_' + super().__str__()
        return s


class Mode(object):
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
            self._logqs = tf.placeholder(tf.float32, [batch_size], name='q_logps')
            #############################################################
            # compute the energy function phi
            #############################################################
            self.net = net.create_net(config, is_training, reuse)
            self.phi, self.vars = self.net.output(self._inputs, self._lengths, reuse=reuse)
            self.var_size = tf.add_n([tf.size(v) for v in self.vars])

            self.logz_a = tf.get_variable(name='logz_a', shape=None, dtype=tf.float32,
                                          initializer=np.array(self.config.init_logz_a, 'float32')
                                          )
            self.logz_b = tf.get_variable(name='logz_b', shape=None, dtype=tf.float32,
                                          initializer=np.array(self.config.init_logz_b, 'float32')
                                          )
            self.logz = self.logz_a * tf.cast(self._lengths, tf.float32) + self.logz_b

            self.pi = tf.constant(self.config.pi, dtype=tf.float32, name='pi')
            self.log_ratio = self.phi + tf.log(tf.gather(self.pi, self._lengths)) - self.logz
            self.logps = self.log_ratio + self._logqs

            self.saver = tf.train.Saver(self.vars)

            ###################################################
            # for NCE training
            ###################################################
            if is_training:
                #############################################################
                # compute the loss
                #############################################################
                # the noise log-probs for all the input sequences, of shape [batch_size]
                self._data_num = tf.placeholder(tf.int32, None, name='data_num')

                data_loss = self.cmp_data_loss(self.log_ratio[0: self._data_num],
                                               tf.zeros_like(self.log_ratio[0: self._data_num]))
                noise_loss = self.cmp_noise_loss(self.log_ratio[self._data_num:],
                                                 tf.zeros_like(self.log_ratio[self._data_num:]))

                data_loss /= tf.cast(self._data_num, tf.float32)
                noise_loss /= tf.cast(tf.shape(self._inputs)[0] - self._data_num, tf.float32)
                self.loss = tf.concat([data_loss, noise_loss], axis=0)

                ###################################
                # training
                ###################################
                total_vars = self.vars + [self.logz_a, self.logz_b]
                print('[%s.%s] variables in %s' % (__name__, self.__class__.__name__, name))
                for v in total_vars:
                    print('\t' + v.name, v.shape, v.device)

                self.trainop = layers.TrainOp(-self.loss, total_vars,
                                              optimize_method=self.config.opt_method,
                                              l2_reg=self.config.l2_reg,
                                              name=name + '/train_op')

                # self.trainop_logz = layers.TrainOp(-self.loss, [self.logz_a, self.logz_b],
                #                                    optimize_method=self.config.opt_method,
                #                                    name=name + '/train_op_logz')

    def cmp_data_loss(self, logp, logq):
        s = self.config.fdiv_type.lower()

        if s == 'gan':
            return logp - layers.logaddexp(logp, logq)
        elif s.find('nce') == 0:
            k = int(s[3:])
            return logp - layers.logaddexp(logp, np.log(k) + logq)
        elif s == 'kl':
            return 1 + logp - logq
        elif s == 'rkl':
            return - tf.exp(logq - logp)
        else:
            raise TypeError('undefined f-Div type=' + self.config.fdiv_type)

    def cmp_noise_loss(self, logp, logq):
        s = self.config.fdiv_type.lower()

        if s == 'gan':
            return logq - layers.logaddexp(logp, logq)
        elif s.find('nce') == 0:
            k = int(s[3:])
            return k * (np.log(k) + logq - layers.logaddexp(logp, np.log(k) + logq))
        elif s == 'kl':
            return -tf.exp(logp - logq)
        elif s == 'rkl':
            return 1 + logq - logp
        else:
            raise TypeError('undefined f-Div type=' + self.config.fdiv_type)

    def get_param_num(self):
        return tf.get_default_session().run(self.var_size)

    def get_logps(self, seq_list, logqs):
        inputs, lengths = reader.produce_data_to_array(seq_list)
        return tf.get_default_session().run(self.logps,
                                            {self._inputs: inputs,
                                             self._lengths: lengths,
                                             self._logqs: logqs})

    # def get_phis(self, seq_list):
    #     inputs, lengths = reader.produce_data_to_array(seq_list)
    #     return tf.get_default_session().run(self.phi,
    #                                         {self._inputs: inputs,
    #                                          self._lengths: lengths})

    def get_logz(self, seq_list):
        inputs, lengths = reader.produce_data_to_array(seq_list)
        return tf.get_default_session().run(self.logz,
                                            {self._inputs: inputs,
                                             self._lengths: lengths})

    def get_logz_vars(self):
        return tf.get_default_session().run([self.logz_a, self.logz_b])

    def update(self, seq_list, data_num, lr=1.0):
        inputs, lengths = reader.produce_data_to_array(seq_list)
        # update parameters
        self.trainop.set_lr(tf.get_default_session(), lr)
        loss = self.trainop.update(tf.get_default_session(), {self._inputs: inputs,
                                                              self._lengths: lengths,
                                                              self._data_num: data_num})

        # # update logz
        # self.trainop_logz.set_lr(tf.get_default_session(), lr_logz)
        # self.trainop_logz.update(tf.get_default_session(), {self._inputs: inputs,
        #                                                     self._lengths: lengths,
        #                                                     self._logqs: logqs,
        #                                                     self._data_num: data_num})

        return loss

    def save(self, fname):
        self.saver.save(tf.get_default_session(), fname + '.ckpt')

    def restore(self, fname):
        self.saver.restore(tf.get_default_session(), fname + '.ckpt')