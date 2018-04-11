import tensorflow as tf
import os

from base import *
from lm import *
from trf.common import feat2 as feat
from trf.sa import net


class BaseConfig(wb.Config):
    def __init__(self, type_name):
        wb.Config.value_encoding_map[lr.LearningRate] = str

        self.type = type_name
        self.L2_reg = 1e-6

    def __str__(self):
        return 'NonePotential'


class FeatConfig(BaseConfig):
    def __init__(self, data):
        super().__init__('feat')

        self.feat_type_file = None
        self.feat_cluster = None

        self.pre_compute_data_exp = True
        self.pre_compute_data_var = False
        self.var_gap = 1e-6

    def __str__(self):
        feat_name = os.path.split(self.feat_type_file)[-1]
        feat_name = os.path.splitext(feat_name)[0]

        if self.feat_cluster is not None:
            feat_name += '_c{}'.format(self.feat_cluster)
        return 'feat{}_L2reg{:.2e}'.format(feat_name, self.L2_reg)


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

    def get_value_depend(self, seq_list, pos):
        return self.get_value(seq_list)

    def initialize(self):
        pass

    def update(self, data_list, data_scalars, sample_list, sample_scalars, learning_rate=1.0):
        return None

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

        if opt_method.lower() == 'adam':
            self.config.pre_compute_data_var = False

        # self.len_factor = self.config.pi_true / self.config.pi_0

        wftype, cftype = feat.separate_type(feat.read_feattype_file(self.config.feat_type_file))
        if self.data.word_to_class is not None:
            wftype.update(cftype)

        # if wb.is_linux():
        #     self.feat = feat.FastFeats(wftype)
        # else:
        self.feat = feat.Feats(wftype)

        self.update_op = None
        self.data_exp = None
        self.data_var = None

    def get_param_num(self):
        return self.feat.num

    def get_params(self):
        return self.feat.values

    def get_value(self, seq_list):
        w = self.feat.seq_list_weight(
            seq.hard_class_seqs(seq_list, self.data.word_to_class)
        )
        return np.array(w)

    def get_value_depend(self, word_list, pos):
        seq_list = seq.hard_class_seqs(word_list, self.data.word_to_class)
        w1 = self.feat.seq_list_weight(seq_list, (0, pos))
        w2 = self.feat.seq_list_weight(seq_list, (1, pos))
        return np.array(w1) + np.array(w2)

    def feat_count(self, cur_feat, seq_list, seq_scalar):
        """
        compute sum_x d * f(x)
        Args:
            cur_feat:
            seq_list:
            seq_scalar:

        Returns:

        """
        buf = np.zeros_like(cur_feat.values)
        a_list = cur_feat.seq_list_find(seq_list)
        for a, d in zip(a_list, seq_scalar):
            for i in a:
                buf[i] += d
        return buf

    def get_exp(self, seq_list, seq_scalar):
        exp = self.feat_count(self.feat,
                              seq.hard_class_seqs(seq_list, self.data.word_to_class),
                              seq_scalar)
        return exp

    def feat_count2(self, cur_feat, seq_list, seq_scalar):
        """
        compute Sum_x d * f(x)^2
        Args:
            cur_feat:
            seq_list:
            seq_scalar:

        Returns:
        """
        buf = np.zeros_like(cur_feat.values)
        a_list = cur_feat.seq_list_find(seq_list)
        for a, d in zip(a_list, seq_scalar):
            f_count = dict()
            for i in a:
                f_count.setdefault(i, 0)
                f_count[i] += 1
            for i, n in f_count.items():
                buf[i] += d * n * n
        return buf

    def get_exp2(self, seq_list, seq_scalar):
        exp = self.feat_count2(self.feat,
                               seq.hard_class_seqs(seq_list, self.data.word_to_class),
                               seq_scalar)
        return exp

    def get_gradient(self, data_list, data_scalar, sample_list, sample_scalar):

        # data_scalar = np.ones(len(data_list)) / len(data_list)
        # sample_len = np.array([len(x) for x in sample_list])
        # sample_scalar = self.len_factor[sample_len - self.config.min_len] / len(sample_list)

        exp_d = self.get_exp(data_list, data_scalar) if self.data_exp is None else self.data_exp
        exp_s = self.get_exp(sample_list, sample_scalar)
        return exp_d - exp_s - self.config.L2_reg * self.get_params()

    def update(self, data_list, data_scalars, sample_list, sample_scalars, learning_rate=1.0):
        exp_d = self.get_exp(data_list, data_scalars) if self.data_exp is None else self.data_exp
        exp_s = self.get_exp(sample_list, sample_scalars)
        g = exp_d - exp_s - self.config.L2_reg * self.get_params()

        g_norm = np.sqrt(np.sum(g * g))
        if self.data_var is not None:
            g /= self.data_var
        d = self.update_op.update(-g, learning_rate)
        self.feat.values += d

        return g_norm

    def initialize(self):
        if self.feat.num == 0:
            with wb.processing('load features ...'):
                self.feat.load_from_seqs(seq.hard_class_seqs(self.data.datas[0], self.data.word_to_class))
        else:
            print('[{}.{}] Features exist. Don\'t reload features'.format(__name__, self.__class__.__name__))

        self.update_op = wb.ArrayUpdate(self.get_param_num(), {'name': self.opt_method})

        if self.config.pre_compute_data_exp:
            with wb.processing('compute feat-exp on data'):
                scalar = np.ones(len(self.data.datas[0])) / len(self.data.datas[0])
                self.data_exp = self.get_exp(self.data.datas[0], scalar)

        if self.config.pre_compute_data_var:
            with wb.processing('compute feat-var on data'):
                self.data_var = self.compute_data_var(self.data.datas[0])
                self.data_var = np.maximum(self.data_var, self.config.var_gap)

    def save(self, fname):
        with open(fname + '.feat', 'wt') as f:
            self.feat.save(f)

    def restore(self, fname):
        with open(fname + '.feat', 'rt') as f:
            self.feat.restore(f)

    def compute_data_var(self, seq_list):

        n = len(seq_list)
        scalar = np.ones(n) / n
        exp = self.get_exp(seq_list, scalar)
        exp2 = self.get_exp2(seq_list, scalar)
        return exp2 - exp ** 2 + self.config.L2_reg


class NetPhi(Base):
    def __init__(self, config, data, opt_method='sgd', device='/gpu:0'):
        self.config = config
        self.data = data

        # revise the opt_method
        config.opt_method = opt_method

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

    def update(self, data_list, data_scalars, sample_list, sample_scalars, learning_rate=1.0):
        return self.train_net.update(data_list, data_scalars, sample_list, sample_scalars, learning_rate=learning_rate)

    def save(self, fname):
        self.saver.save(tf.get_default_session(), fname + '.ckpt')

    def restore(self, fname):
        self.saver.restore(tf.get_default_session(), fname + '.ckpt')


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
        self.update_op = wb.ArrayUpdate(self.zeta, {'name': self.opt_method})

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