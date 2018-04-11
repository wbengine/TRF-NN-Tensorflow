import numpy as np
import os

from base import *
from lm import *
from trf.common import *
from trf.sa import simulater, pot, trf


class DefaultOps(trf.DefaultOps):
    pass


class Config(trf.Config):
    def __init__(self, data):
        super().__init__(data)

        self.eps_min = 0.5
        self.eps_max = 1.5
        self.Lstep = 10

    def __str__(self):
        return super().__str__() + '_2'


class TRF(trf.TRF):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):
        super().__init__(config, data, logdir, device, name)

        self.mass = np.ones(config.max_len+1)
        self.eps_center = (self.config.eps_min + self.config.eps_max)/2
        self.eps_width = self.config.eps_max - self.eps_center

    def count_laplace_mass(self):
        exp = np.zeros(self.data.get_max_len() + 1)
        var = np.zeros_like(exp)
        pos_count = np.zeros_like(exp)

        for seq in self.data.datas[0]:
            n = len(seq)
            pos_count[0:n] += 1

        for seq in self.data.datas[0]:
            n = len(seq)
            exp[0: n] += np.array(seq) / pos_count[0: n]

        for seq in self.data.datas[0]:
            n = len(seq)
            var[0: n] += (np.array(seq) - exp[0: n]) ** 2 / pos_count[0: n]

        self.mass = np.maximum(1, np.sqrt(var))
        print('[{}.{}] mass={}'.format(__name__, self.__class__.__name__, self.mass))

    def initialize(self):
        super().initialize()

        # self.count_laplace_mass()

    def coord_integrator(self, x_f, p, i, eps):
        self.mv_times += 1

        x_f = np.array(x_f, dtype='float32')
        x_new = np.array(x_f, dtype='float32')

        x_new[i] += eps / self.mass[i] * np.sign(p[i])
        if x_new[i] < 0 or x_new[i] > self.config.vocab_size:
            p[i] = - p[i]
        else:
            logps = self.get_log_probs([x_f.astype('int32').tolist(), x_new.astype('int32').tolist()], for_eval=False)
            dU = -logps[1] + logps[0]
            if np.abs(p[i]) / self.mass[i] > dU:
                self.mv_success += 1
                x_f = x_new
                p[i] -= self.mass[i] * dU
            else:
                p[i] = - p[i]
        return x_f, p

    def coord_integrator_batch(self, input_xf, input_n, input_p, i, eps, input_u):
        """

        Args:
            input_xf: array, (chain_num, max_len)
            input_n:  array, (chain_num,)
            input_p:  array, (chain_num, max_len)
            i:  integer
            eps: array, (chain_num,)
            input_U:  array, (chain_num,)

        Returns:
            next_xf, next_p
        """
        input_xf = input_xf.astype('float32')
        next_xf = np.array(input_xf, dtype='float32')
        next_xf[:, i] += eps / self.mass[i] * np.sign(input_p[:, i])

        legal_value_flag = np.logical_and(next_xf[:, i] >= 0, next_xf[:, i] <= self.config.vocab_size)
        legal_value_idx = np.where(legal_value_flag)[0]

        next_u = np.ones_like(input_u) * 1e3
        if len(legal_value_idx) > 0:
            next_u_legal = - self.logps(next_xf[legal_value_idx].astype('int32'), input_n[legal_value_idx], for_eval=False)
            next_u[legal_value_idx] = next_u_legal
        dU = next_u - input_u

        update_flag = np.abs(input_p[:, i]) / self.mass[i] > dU
        update_flag = np.logical_and(update_flag, legal_value_flag)

        res_x = np.array(input_xf)
        res_p = np.array(input_p)
        for k, is_update in enumerate(update_flag):
            if is_update:
                res_x[k] = next_xf[k]
                res_p[k, i] = input_p[k, i] - self.mass[i] * dU[k]
            else:
                res_p[k, i] = - input_p[k, i]

        res_u = np.where(update_flag, next_u, input_u)

        revise_value_flag = res_x[:, i].astype('int32') != input_xf[:, i].astype('int32')

        self.mv_times += len(update_flag)
        self.mv_success += np.sum(revise_value_flag)

        out_line = '[Markov move] is_update={}/{}'.format(np.sum(update_flag), len(update_flag)) + \
                   'is_revise={}/{}'.format(np.sum(revise_value_flag), len(revise_value_flag)) + \
                   ' p=' + log.to_str(input_p[:, i]) + \
                   ' dU=' + log.to_str(dU) + \
                   ' eps=' + log.to_str(eps)
        f = self.write_files.get('markov')
        f.write(out_line + '\n')
        f.flush()

        return res_x, res_p, res_u

    def markov_move(self, x):
        p = [np.random.laplace(0, self.mass[i]) for i in range(len(x))]

        n = len(x)
        x_f = np.array(x, dtype='float32')
        for L in range(self.config.Lstep):
            for i in range(1, n-1):
                x_f, p = self.coord_integrator(x_f, p, i, np.random.uniform(self.config.eps_min, self.config.eps_max))

        return x_f.tolist()

    def markov_move_batch(self, input_xf, input_n):
        # x_list = reader.extract_data_from_array(input_x, input_n)
        # y_list = []
        # for x in x_list:
        #     y_list.append(self.markov_move(x))
        # return reader.produce_data_to_array(y_list, dtype='float32')
        chain_num = input_xf.shape[0]
        max_len = input_xf.shape[1]
        input_p = [np.random.laplace(0, self.mass[i], size=(chain_num, 1)) for i in range(max_len)]
        input_p = np.concatenate(input_p, axis=1).astype('float32')

        input_xf = np.array(input_xf, dtype='float32')
        input_u = - self.logps(input_xf.astype('int32'), input_n, for_eval=False)
        for L in range(self.config.Lstep):
            for i in range(1, max_len-1):
                idx = np.where(input_n-1 > i)[0]

                eps = np.random.uniform(max(0, self.eps_center - self.eps_width),
                                        self.eps_center + self.eps_width, size=len(idx))
                res_x, res_p, res_u = self.coord_integrator_batch(input_xf[idx], input_n[idx], input_p[idx], i,
                                                                  eps, input_u[idx])

                input_xf[idx] = res_x
                input_p[idx] = res_p
                input_u[idx] = res_u

        return input_xf, input_n

    def sample(self, input_xf, input_n):

        with self.time_recoder.recode('local_jump'):
            next_x, next_n = self.local_jump_batch(input_xf.astype('int32'), input_n)

            next_xf = []
            for old_xf, new_x in zip(reader.extract_data_from_array(input_xf, input_n),
                                     reader.extract_data_from_array(next_x, next_n)):
                if len(old_xf) == len(new_x):
                    next_xf.append(old_xf)
                elif len(new_x) > len(old_xf):
                    next_xf.append(old_xf[0:-1] + new_x[len(old_xf)-1:])
                else:
                    next_xf.append(old_xf[0: len(new_x)-1] + new_x[-1:])

            input_xf, input_n = reader.produce_data_to_array(next_xf, dtype='float32')
            assert np.all(input_n == next_n)

        with self.time_recoder.recode('markov_move'):
            input_xf, input_n = self.markov_move_batch(input_xf, input_n)

        return input_xf, input_n

    def draw(self, n):
        """
        calling self.sample to draw n samples

        Args:
            n: the sample numbers

        Returns:
            a list of n sequences
        """
        self.lj_times = 0
        self.lj_success = 0
        self.mv_times = 0
        self.mv_success = 0

        seq_list = []
        for i in range(n//self.config.chain_num):
            self.sample_seq = self.sample(*self.sample_seq)
            seq_list += reader.extract_data_from_array(self.sample_seq[0].astype('int32'),
                                                       self.sample_seq[1])   # copy the sequence

        if self.lj_times > 0:
            self.lj_rate = 0.9 * self.lj_rate + 0.1 * (self.lj_success / self.lj_times)
        if self.mv_times > 0:
            self.mv_rate = 0.9 * self.mv_rate + 0.1 * (self.mv_success / self.mv_times)

        # if self.mv_rate < 0.5 and self.eps_center > 0.1:
        #     self.eps_center /= 2

        with self.time_recoder.recode('write_sample'):
            f = self.write_files.get('sample')
            for x in seq_list:
                log.write_seq(f, x)

            f = self.write_files.get('mcmc')
            for x, n in self.sample_mcmc:
                seqs = reader.extract_data_from_trf(x, n)
                f.write('\n'.join([str(a) for a in seqs]) + '\n')
            f.flush()

        return seq_list



