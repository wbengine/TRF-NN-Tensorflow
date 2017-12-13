import tensorflow as tf
import os

from base import *
from lm import *
from trf.common import feat
from trf.nce import net, noise


class BaseConfig(wb.Config):
    def __init__(self, type_name):
        wb.Config.value_encoding_map[lr.LearningRate] = str

        self.type = type_name

    def __str__(self):
        return 'NonePotential'


class FeatConfig(BaseConfig):
    def __init__(self, data):
        super().__init__('feat')

        self.feat_type_file = None
        self.feat_cluster = None
        self.var_gap = 1e-3

    def __str__(self):
        feat_name = os.path.split(self.feat_type_file)[-1]
        feat_name = os.path.splitext(feat_name)[0]
        return 'feat{}'.format(feat_name)


class NetConfig(BaseConfig, net.Config):
    def __init__(self, data):
        BaseConfig.__init__(self, 'net')
        net.Config.__init__(self, data)

    def __str__(self):
        return net.Config.__str__(self)


class Base(object):
    def get_param_num(self):
        return 0

    def get_value(self, seq_list):
        return np.zeros(len(seq_list))

    def get_value_for_train(self, seq_list):
        return np.zeros(len(seq_list))

    def initialize(self):
        pass

    def update(self, seq_list, cluster_weights, cluster_m=None, learning_rate=1.0):
        pass

    def save(self, fname):
        pass

    def restore(self, fname):
        pass

    @property
    def global_step(self):
        return None


class FeatPhi(Base):
    def __init__(self, config, data, opt_method):
        self.config = config
        self.data = data
        self.opt_method = opt_method

        wftype, cftype = feat.separate_type(feat.read_feattype_file(self.config.feat_type_file))
        self.wfeat = feat.Feats(wftype)
        # self.cfeat = feat.Feats(cftype)

        self.update_op = None

    def get_param_num(self):
        return self.wfeat.num

    def get_value(self, seq_list):
        return self.wfeat.seq_list_weight(seq_list)

    def get_value_for_train(self, seq_list):
        return self.get_value(seq_list)

    def get_gradient(self, seq_list, cluster_weights):
        grad = np.zeros_like(self.wfeat.values)
        for x, w in zip(seq_list, cluster_weights):
            for i in self.wfeat.seq_find(x):
                grad[i] += w
        return grad

    def get_update(self, seq_list, cluster_weights, cluster_m):
        grad = np.zeros_like(self.wfeat.values)
        sigma = np.zeros_like(self.wfeat.values)
        for x, w, m in zip(seq_list, cluster_weights, cluster_m):
            a = self.wfeat.seq_find(x)   # find the features
            for i in a:
                grad[i] += w

            c = dict()  # count the feature
            for i in a:
                if i in c:
                    c[i] += 1
                else:
                    c[i] = 1
            for i, n in c.items():
                sigma[i] += m * (n**2)

        return grad / np.maximum(sigma, self.config.var_gap)

    def update(self, seq_list, cluster_weights, cluster_m=None, learning_rate=1.0):
        # if cluster_m is None:
        g = self.get_gradient(seq_list, cluster_weights)
        # else:
        #     g = self.get_update(seq_list, cluster_weights, cluster_m)
        self.wfeat.values += self.update_op.update(-g, learning_rate)

    def initialize(self):
        self.wfeat.load_from_seqs(self.data.datas[0])
        self.update_op = wb.ArrayUpdate(self.wfeat.values, {'name': self.opt_method})

    def save(self, fname):
        with open(fname + '.feat', 'wt') as f:
            self.wfeat.save(f)

    def restore(self, fname):
        with open(fname + '.feat', 'rt') as f:
            self.wfeat.restore(f)


class NetPhi(Base):
    def __init__(self, config, data, opt_method='sgd', device='/gpu:0'):
        self.config = config
        self.data = data

        # revise the opt_method
        config.opt_method = opt_method

        # self.update_state = config.structure_type == 'rnn' and config.rnn_type == 'lstm'
        # self.update_state = False
        # print('[{}.NetPhi] update_stata={}'.format(__name__, self.update_state))

        self.train_net = net.NetBase(config, is_training=True, device=device, name='trfnet', reuse=None)
        self.eval_net = net.NetBase(config, is_training=False, device=device, name='trfnet', reuse=True)

        self.saver = tf.train.Saver(self.train_net.vars)

    def get_param_num(self):
        return self.train_net.get_param_num()

    def get_value(self, seq_list):
        return self.eval_net.get_phi(seq_list)

    def get_value_for_train(self, seq_list):
        return self.train_net.get_phi(seq_list)

    def initialize(self):
        pass

    def update(self, seq_list, cluster_weights, cluster_m=None, learning_rate=1.0):
        self.train_net.update(seq_list, cluster_weights, cluster_m, learning_rate)
        # # replace the state
        # if self.update_state:
        #     self.train_net.fstate = self.train_net.fstate2

    def save(self, fname):
        self.saver.save(tf.get_default_session(), fname + '.ckpt')

    def restore(self, fname):
        self.saver.restore(tf.get_default_session(), fname + '.ckpt')

    @property
    def global_step(self):
        return self.train_net.trainop.global_step


class Norm(object):
    def __init__(self, config, data, opt_method='sgd'):
        self.config = config
        self.data = data
        self.opt_method = opt_method

        self.logz = np.concatenate([[0]*self.config.min_len, self.config.init_logz])
        self.update_op = wb.ArrayUpdate(self.logz, {'name': self.opt_method})

    def get_logz(self, lengths=None):
        if lengths is None:
            return self.logz
        return self.logz[lengths]

    def get_var(self):
        return self.logz

    def get_gradient(self, seq_list, cluster_weights):
        grad = np.zeros_like(self.logz)
        for x, w in zip(seq_list, cluster_weights):
            grad[len(x)] += w
        return grad

    def get_variance(self, seq_list, cluster_m):
        sigma = np.zeros_like(self.logz)
        for x, m in zip(seq_list, cluster_m):
            sigma[len(x)] += m
        return sigma

    def update(self, seq_list, cluster_weights, cluster_m=None, learning_rate=1.0):
        g = self.get_gradient(seq_list, cluster_weights)
        if cluster_m is not None:
            sigma = self.get_variance(seq_list, cluster_m)
            g /= np.maximum(sigma, self.config.var_gap)

        self.logz += self.update_op.update(g, learning_rate)

    def save(self, fname):
        with open(fname, 'wt') as f:
            f.write('len\tlogz\n')
            for i, v in enumerate(self.logz):
                f.write('{}\t{}\n'.format(i, v))

    def restore(self, fname):
        with open(fname, 'rt') as f:
            f.readline()
            lens = []
            logz = []
            for line in f:
                a = line.split()
                lens.append(int(a[0]))
                logz.append(float(a[1]))
            self.logz = np.array(logz)


class NormLinear(object):
    """
    define the normalization constants logZ_l = a + b * l, l = 0, 2, ..., m-1
    """
    def __init__(self, config, data, opt_method='sgd'):
        self.config = config
        self.data = data
        self.opt_method = opt_method

        self.var_a = config.init_logz[0]
        self.var_b = config.init_logz[1]

        self.update_op = wb.ArrayUpdate([self.var_a, self.var_b], {'name': self.opt_method})

    def get_logz(self, lengths=None):
        if lengths is None:
            # return a logz array
            lengths = np.arange(0, self.config.max_len+1, dtype='int32')
        return self.var_a + self.var_b * (lengths - 1)

    def get_var(self):
        return [self.var_a, self.var_b]

    def get_gradient(self, seq_list, cluster_weights):
        """
        g_a = \sum_x w(x)
        g_b = \sum_x w(x) (l(x) - l_min)
        Args:
            seq_list:
            cluster_weights:

        Returns:

        """
        lengths = np.array([len(x) for x in seq_list])
        grad_a = np.sum(cluster_weights)
        grad_b = np.sum(cluster_weights * (lengths - 1))
        return np.array([grad_a, grad_b])

    def get_variance(self, seq_list, cluster_m):
        sigma = np.zeros_like(self.logz)
        for x, m in zip(seq_list, cluster_m):
            sigma[len(x)] += m
        return sigma

    def update(self, seq_list, cluster_weights, cluster_m=None, learning_rate=1.0):
        g = self.get_gradient(seq_list, cluster_weights)
        # if cluster_m is not None:
        #     sigma = self.get_variance(seq_list, cluster_m)
        #     g /= np.maximum(sigma, self.config.var_gap)

        d = self.update_op.update(g, learning_rate)
        self.var_a += d[0]
        self.var_b += d[1]

    def save(self, fname):
        with open(fname, 'wt') as f:
            f.write('zeta_a={}\n'.format(self.var_a))
            f.write('zeta_b={}\n'.format(self.var_b))
            f.write('len\tlogz\n')
            for i in range(self.config.min_len, self.config.max_len+1):
                f.write('{}\t{}\n'.format(i, self.get_logz(i)))

    def restore(self, fname):
        with open(fname, 'rt') as f:
            self.var_a = float(f.readline().split('=')[-1])
            self.var_b = float(f.readline().split('=')[-1])
