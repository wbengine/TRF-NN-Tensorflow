import numpy as np

from base import *


class NormBase(object):
    def get_logz(self, lengths=None):
        pass

    def set_logz1(self, logz1):
        pass

    def update(self, seq_list, scalar=None, learning_rate=1.0):
        pass

    def save(self, fname):
        pass

    def restore(self, fname):
        pass


class Norm(NormBase):
    def __init__(self, config, data, opt_method='sgd'):
        self.config = config
        self.data = data
        self.opt_method = opt_method

        self.zeta = np.array(self.config.init_logz) - self.config.init_logz[0]
        self.logz1 = self.config.init_logz[0]
        # self.update_op = wb.ArrayUpdate(self.zeta, {'name': self.opt_method})

    def get_logz(self, lengths=None):
        if lengths is None:
            return self.zeta + self.logz1
        return self.zeta[np.array(lengths) - self.config.min_len] + self.logz1

    def set_logz1(self, logz1):
        self.logz1 = logz1

    def set_logz(self, logz):
        """logz[config.min_len: config.max_len+1]"""
        self.logz1 = logz[0]
        self.zeta = logz - logz[0]

    def get_gradient(self, sample_list, scalar):

        if scalar is None:
            scalar = [1.0 / len(sample_list)] * len(sample_list)

        grad = np.zeros_like(self.zeta)
        for x, s in zip(sample_list, scalar):
            grad[len(x) - self.config.min_len] += s
        # grad /= len(sample_list)
        grad /= self.config.pi_0[self.config.min_len:]
        return grad

    def update(self, seq_list, scalar=None, learning_rate=1.0):
        g = self.get_gradient(seq_list, scalar)

        self.zeta += np.clip(learning_rate * g, a_min=0, a_max=self.config.zeta_gap)
        self.zeta -= self.zeta[0]

        return np.sqrt(np.sum(g * g))

    def save(self, fname):
        with open(fname, 'wt') as f:
            f.write('logz1={}\n'.format(self.logz1))
            f.write('len\tzeta\n')
            for i, v in enumerate(self.zeta):
                f.write('{}\t{}\n'.format(i + self.config.min_len, v))

    def restore(self, fname):
        with open(fname, 'rt') as f:
            self.logz1 = float(f.readline().split('=')[-1])
            f.readline()
            lens = []
            zeta = []
            for line in f:
                a = line.split()
                lens.append(int(a[0]))
                zeta.append(float(a[1]))
            self.zeta = np.array(zeta)


class NormLinear(Norm):
    def __init__(self, config, data, opt_method='sgd'):
        super().__init__(config, data, opt_method)

        self.a = self.zeta[1] - self.zeta[0]
        self.zeta = np.zeros_like(self.zeta)

        self.zeta_update = wb.ArrayUpdate(self.zeta, {'name': opt_method})
        self.a_update = wb.ArrayUpdate(self.a, {'name': opt_method, 'max_norm': 1.0})

    def get_logz(self, lengths=None):
        if lengths is None:
            lengths = np.arange(self.config.min_len, self.config.max_len+1)

        idxs = np.array(lengths) - self.config.min_len
        return self.zeta[idxs] + self.a * idxs + self.logz1

    def get_gradient(self, sample_list, scalar):
        if scalar is None:
            scalar = np.array([1.0 / len(sample_list)] * len(sample_list))

        grad_zeta = np.zeros_like(self.zeta)
        for x, s in zip(sample_list, scalar):
            grad_zeta[len(x) - self.config.min_len] += s
        grad_zeta -= self.config.pi_0[self.config.min_len:]

        grad_a = np.sum(grad_zeta * np.arange(len(grad_zeta)))

        return grad_zeta, grad_a

    def update(self, seq_list, scalar=None, learning_rate=1.0):

        grad_zeta, grad_a = self.get_gradient(seq_list, scalar)

        self.zeta += self.zeta_update.update(-grad_zeta, learning_rate)
        self.zeta -= self.zeta[0]
        self.a += self.a_update.update(-grad_a, learning_rate)

        return np.sqrt(np.sum(grad_zeta ** 2)), np.sqrt(np.sum(grad_a ** 2))

    def save(self, fname):
        with open(fname, 'wt') as f:
            f.write('logz1={}\n'.format(self.logz1))
            f.write('a={}\n'.format(self.a))
            f.write('len\tzeta\n')
            for i, v in enumerate(self.zeta):
                f.write('{}\t{}\n'.format(i + self.config.min_len, v))

    def restore(self, fname):
        with open(fname, 'rt') as f:
            self.logz1 = float(f.readline().split('=')[-1])
            self.a = float(f.readline().split('=')[-1])
            f.readline()
            lens = []
            zeta = []
            for line in f:
                a = line.split()
                lens.append(int(a[0]))
                zeta.append(float(a[1]))
            self.zeta = np.array(zeta)




