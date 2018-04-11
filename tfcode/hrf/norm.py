
from base import *


class NormBase(object):
    def get_logz(self, lengths=None):
        pass

    def set_logz1(self, logz1):
        pass

    def update(self, seq_list, scalar, learning_rate=1.0):
        pass

    def save(self, fname):
        pass

    def restore(self, fname):
        pass


class NormFixed(NormBase):
    def __init__(self, config, data, opt_method='sgd'):
        self.config = config
        self.data = data
        self.opt_method = opt_method
        self.logz1 = self.config.init_logz[0]

    def get_logz(self, lengths=None):
        if lengths is None:
            lengths = np.linspace(self.config.min_len, self.config.max_len,
                                  self.config.max_len - self.config.min_len + 1)
        return self.logz1 * (lengths - self.config.min_len + 1)

    def set_logz1(self, logz1):
        self.logz1 = logz1

    def update(self, seq_list, learning_rate=1.0):
        pass

    def save(self, fname):
        with open(fname, 'wt') as f:
            f.write('logz1={}\n'.format(self.logz1))

    def restore(self, fname):
        with open(fname, 'rt') as f:
            self.logz1 = float(f.readline().split('=')[-1])


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

        # self.zeta += np.clip(learning_rate * g, a_min=0, a_max=self.config.zeta_gap)
        self.zeta += learning_rate * g
        self.zeta -= self.zeta[0]

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


class Norm2(NormBase):
    def __init__(self, trf_g_config, data=None, opt_method='sgd'):
        self.config = trf_g_config
        self.opt_method = opt_method

        self.zeta = np.array(self.config.init_logz)
        self.logz = trf_g_config.global_logz
        self.sigma = np.sqrt(self.config.pi_true[self.config.min_len:] -
                             np.power(self.config.pi_true[self.config.min_len:], 2))

    def get_logz(self, lengths=None):
        if lengths is None:
            return self.zeta + self.logz
        return self.zeta[np.array(lengths) - self.config.min_len] + self.logz

    def get_gradient(self, sample_list, scaler):
        grad = np.zeros_like(self.zeta)
        for x, s in zip(sample_list, scaler):
            grad[len(x) - self.config.min_len] += s
        grad /= len(sample_list)
        grad -= self.config.pi_true[self.config.min_len:]  # -\pi_data + \pi_sample
        grad /= np.maximum(self.sigma, 1e-6)
        return grad

    def update(self, seq_list, scaler, learning_rate=1.0):
        g = self.get_gradient(seq_list, scaler)

        self.zeta += learning_rate * np.clip(g, a_min=-self.config.zeta_gap, a_max=self.config.zeta_gap)

    def save(self, fname):
        with open(fname, 'wt') as f:
            f.write('logz={}\n'.format(self.logz))
            f.write('len\tzeta\n')
            for i, v in enumerate(self.zeta):
                f.write('{}\t{}\n'.format(i + self.config.min_len, v))

    def restore(self, fname):
        with open(fname, 'rt') as f:
            self.logz = float(f.readline().split('=')[-1])
            f.readline()
            lens = []
            zeta = []
            for line in f:
                a = line.split()
                lens.append(int(a[0]))
                zeta.append(float(a[1]))
            self.zeta = np.array(zeta)


class NormOne(NormBase):
    def __init__(self, trf_g_config):
        super().__init__()

        self.logz = trf_g_config.global_logz

    def get_logz(self, lengths=None):
        return self.logz

    def save(self, fname):
        with open(fname, 'wt') as f:
            f.write('logz={}\n'.format(self.logz))

    def restore(self, fname):
        with open(fname, 'wt') as f:
            self.logz = float(f.readline().split('=')[-1])


class NormLinear(NormBase):
    def __init__(self, config, data, opt_method='sgd'):
        self.config = config
        self.data = data
        self.opt_method = opt_method

        self.logz1 = self.config.init_logz[0]
        self.logz_inc = self.config.init_logz[1] - self.config.init_logz[0]

        self.grad_const = np.dot(self.config.pi_true[self.config.min_len: self.config.max_len+1],
                                 np.arange(0, self.config.max_len+1-self.config.min_len))

        self.update_op = wb.ArrayUpdate([self.logz_inc], {'name': self.opt_method})

    def get_logz(self, lengths=None):
        if lengths is None:
            lengths = list(range(self.config.min_len, self.config.max_len+1))
        return self.logz1 + (np.array(lengths) - self.config.min_len) * self.logz_inc

    def set_logz1(self, logz1):
        self.logz1 = logz1

    def get_gradient(self, sample_list, scalar):

        if scalar is None:
            scalar = [1.0 / len(sample_list)] * len(sample_list)

        grad = 0
        for x, s in zip(sample_list, scalar):
            grad += s * (len(x) - self.config.min_len)
        # grad /= len(sample_list)
        grad -= self.grad_const
        return grad

    def update(self, seq_list, scalar=None, learning_rate=1.0):
        g = self.get_gradient(seq_list, scalar)

        self.logz_inc += self.update_op.update([-g], learning_rate)[0]

        return g

    def save(self, fname):
        with open(fname, 'wt') as f:
            f.write('logz1={}\n'.format(self.logz1))
            f.write('logz_inc={}\n'.format(self.logz_inc))
            f.write('len\tzeta\n')
            for i, v in enumerate(self.get_logz()):
                f.write('{}\t{}\n'.format(i + self.config.min_len, v))

    def restore(self, fname):
        with open(fname, 'rt') as f:
            self.logz1 = float(f.readline().split('=')[-1])
            self.logz_inc = float(f.readline().split('=')[-1])



