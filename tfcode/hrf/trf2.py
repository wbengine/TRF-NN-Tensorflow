import tensorflow as tf
import os
import json
import time
from copy import deepcopy
from collections import OrderedDict
from base import *
from lm import *


from . import crf, pot, tagphi, mixphi, trf, mcmc


class Config(trf.Config):
    def __init__(self, data):
        super().__init__(data)

        self.chain_num = 100
        self.multiple_trial = 1
        self.sample_sub = 1

    def __str__(self):
        s = super().__str__()
        return s.replace('hrf_', 'hrf2_')


class DefaultOps(trf.DefaultOps):
    pass


class TRF(trf.TRF):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):

        super().__init__(config, data, logdir, device, name)

        self.mcmc.fun_logps = self.sample_logps
        self.mcmc.fun_propose_tag = None

    def sample_logps(self, seq_list):
        wod_seqs = [s.x[0] for s in seq_list]
        return super().logpxs(wod_seqs, is_norm=True, for_eval=False)

    def sample(self, seq_list, states_list):
        with self.time_recoder.recode('local_jump'):
            seq_list, states_list = self.mcmc.local_jump(seq_list, states_list)

        with self.time_recoder.recode('markov_move'):
            seq_list, states_list = self.mcmc.markov_move(seq_list, states_list)

        with self.time_recoder.recode('decode'):
            t_list, _ = self.get_tag([s.x[0] for s in seq_list])
            for s, t in zip(seq_list, t_list):
                s.x[1] = t

        return seq_list, states_list

    def eval(self, seq_list, for_eval=True):
        logps = self.get_logpxs(seq.get_x(seq_list), for_eval=for_eval)
        nll = -np.mean(logps)
        words = np.sum([len(x)-1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)
        return nll, ppl

    def update(self, data_list, sample_list):
        # compute the scalars
        data_scalar = np.ones(len(data_list)) / len(data_list)
        sample_len = np.array([len(x) for x in sample_list])
        sample_facter = np.array(self.config.pi_true[self.config.min_len:]) / \
                        np.array(self.config.pi_0[self.config.min_len:])
        sample_scalar = sample_facter[sample_len - self.config.min_len] / len(sample_list)

        # update phi
        with self.time_recoder.recode('update_word'):
            self.phi_word.update(data_list, data_scalar, sample_list, sample_scalar,
                                 learning_rate=self.cur_lr_word)

        if not self.config.fix_crf_model:
            sample_x_list = [s.x[0] for s in sample_list]
            with self.time_recoder.recode('update_marginal'):
                sample_fp_logps_list = self.marginal_logps(sample_x_list)

            with self.time_recoder.recode('update_tag'):
                self.phi_tag.update(data_list, data_scalar, sample_list, sample_scalar,
                                    sample_fp_logps_list=sample_fp_logps_list,
                                    learning_rate=self.cur_lr_tag)

            with self.time_recoder.recode('update_mix'):
                self.phi_mix.update(data_list, data_scalar, sample_list, sample_scalar,
                                    sample_fp_logps_list=sample_fp_logps_list,
                                    learning_rate=self.cur_lr_mix)

        # update zeta
        self.norm_const.update(sample_list, learning_rate=self.cur_lr_logz)
        self.norm_const.set_logz1(self.get_true_logz(self.config.min_len)[0])

        # update simulater
        with self.time_recoder.recode('update_simulater'):
            self.mcmc.update(sample_list)

        # update dbg info
        self.sample_cur_pi.fill(0)
        for x in sample_list:
            self.sample_cur_pi[len(x)] += 1
        self.sample_acc_count += self.sample_cur_pi
        self.sample_cur_pi /= self.sample_cur_pi.sum()

        return None
