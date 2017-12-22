import tensorflow as tf
import numpy as np

from base import *


class BNCELoss(object):
    def __init__(self, inputs, labels, vocab_size, noise_probs, logZ, name='B-NCE-loss'):
        self.vocab_size = vocab_size
        input_shape = tf.shape(inputs)[0:-1]
        input_dim = inputs.shape[-1].value
        B = tf.size(labels)
        # denote:
        #   B = batch_size * step_size = size(labels)

        # smooth noise_probs
        if np.min(noise_probs) == 0:
            print('[%s.%s] smooth noise prob.' % (__name__, self.__class__.__name__))
            noise_probs += 1e-5
            noise_probs /= np.sum(noise_probs)
        elif np.min(noise_probs) < 0:
            print('[%s.%s] fix noise prob.' % (__name__, self.__class__.__name__))
            noise_probs += -np.min(noise_probs) + 1e-5
            noise_probs /= np.sum(noise_probs)

        with tf.name_scope(name):
            # define the noise logprobs
            noise_logps = tf.get_variable(name + '/noise_logp', dtype=tf.float32,
                                          initializer=tf.constant(np.log(noise_probs), dtype=tf.float32),
                                          trainable=False)
            # define the softmax W and b
            softmax_w = tf.get_variable(name + '/w', [input_dim, vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable(name + '/b', [vocab_size], dtype=tf.float32)

            labels = tf.reshape(labels, [-1])
            w_cur = tf.transpose(tf.gather(tf.transpose(softmax_w), labels))  # [input_dim, B]
            b_cur = tf.gather(softmax_b, labels)                              # [B]

            logout = tf.reshape(inputs, [-1, input_dim])        # [B, input_dim]
            logout = tf.matmul(logout, w_cur) + b_cur          # [B, B]
            logout = logout - logZ                             # [B, B]

            logpn = tf.reshape(tf.gather(noise_logps, labels), [1, -1])  # [1, B]
            logpn = tf.tile(logpn, [B, 1])             # [B, B]
            logpn_times = tf.log(tf.cast(B-1, dtype=tf.float32)) + logpn
            logY = layers.logaddexp(logout, logpn_times)

            cluster_logp = logout - logY
            loss_diag = cluster_logp * tf.eye(B)
            loss_other = (logpn_times - logY) * (1. - tf.eye(B))
            loss_all = -(loss_diag + loss_other)
            self._loss = tf.reduce_sum(loss_all) / tf.cast(B, dtype=tf.float32)

            # the logprobs
            self._logps = tf.matmul(logout * tf.eye(B), tf.ones([B, 1], dtype=tf.float32))
            self._logps = tf.reshape(self._logps, input_shape)
            self._probs = tf.exp(self._logps)

            # dbgs
            truelogZ = tf.matmul(tf.reshape(inputs, [-1, input_dim]), softmax_w) + softmax_b  # [B, vocab_size]
            truelogZ = tf.reduce_logsumexp(truelogZ, axis=-1)  # [B]
            self._dbgs = {'loss_all': loss_all,
                          'loss': self._loss,
                          'P(C)': cluster_logp,
                          'B': B,
                          'truelogZ': truelogZ,
                          'truelogZ_mean': tf.reduce_mean(truelogZ),
                          'logO': logout,
                          'logY': logY,
                          'labels': labels,
                          'logpn': logpn,
                          'logpm': self._logps}

    @property
    def loss(self):
        return self._loss

    @property
    def logps(self):
        return self._logps

    @property
    def probs(self):
        return self._probs

    @property
    def dbgs(self):
        return self._dbgs
