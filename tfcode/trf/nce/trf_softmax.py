import numpy as np

from base import *
from . import trf
from .trf import DefaultOps


class Config(trf.Config):
    def __str__(self):
        s = super().__str__()
        return s.replace('_nce', '_snce')


class TRF(trf.TRF):
    def cmp_cluster_weight(self, logpm, logpn, data_num, lengths):
        w_data = np.ones(data_num) / data_num
        w_noise = - np.exp(logpm[data_num:] - logsumexp(logpm))
        return np.concatenate([w_data, w_noise])

    def cmp_cluster_loss(self, logpm, logpn, data_num, lengths):
        loss_data = logpm[0:data_num] - logsumexp(logpm)
        loss_noise = np.zeros_like(logpm[data_num:])
        return -np.concatenate([loss_data, loss_noise])

    def cmp_cluster_logps(self, logpm, logpn):
        return logpm - logsumexp(logpm)