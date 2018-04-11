import os

import tensorflow as tf

from base import *
from lm import *
from trf.common import feat2 as feat
# from . import wodnet
from trf.sa import net as wodnet


def create(word_config, data_seq_list, opt_method, device):
    if word_config is None:
        return Base()
    elif isinstance(word_config, FeatConfig):
        return FeatPhi(word_config, data_seq_list, opt_method)
    elif isinstance(word_config, NetConfig):
        return NetPhi(word_config, data_seq_list, opt_method, device)
    else:
        raise TypeError('[{}] create: undefined config type={}'.format(__name__, type(word_config)))


class BaseConfig(wb.Config):
    def __init__(self):
        pass

    def __str__(self):
        return 'NonePotential'


class FeatConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self.feat_dict = {'w[1:2]': 0}
        # self.feat_cluster = None
        self.L2_reg = 1e-6

        self.pre_compute_data_exp = False
        self.pre_compute_data_var = False
        self.var_gap = 1e-6

    def __str__(self):
        max_order = feat.Feats(self.feat_dict).get_order()
        return 'w{}g'.format(max_order)


class NetConfig(BaseConfig, wodnet.Config):
    def __init__(self, data):
        BaseConfig.__init__(self)
        wodnet.Config.__init__(self, data)

    def __str__(self):
        return wodnet.Config.__str__(self)


class Base(object):
    def get_param_num(self):
        return 0

    def get_value(self, seq_list, depend_on=None):
        return np.zeros(len(seq_list))

    def set_params(self, value=None):
        pass

    def get_order(self, level=None):
        return 0

    def initialize(self):
        pass

    def update(self, data_list, data_scalars, sample_list, sample_scalars, learning_rate=1.0):
        pass

    def save(self, fname):
        pass

    def restore(self, fname):
        pass

    @property
    def global_step(self):
        return None


class FeatPhi(Base):
    def __init__(self, config, data_seq_list, opt_method):
        self.config = config
        self.data_seq_list = data_seq_list
        self.opt_method = opt_method

        if opt_method.lower() == 'adam':
            self.config.pre_compute_data_var = False

        # if wb.is_linux():
        #     self.feats = feat.FastFeats(self.config.feat_dict)
        # else:
        self.feats = feat.Feats(self.config.feat_dict)

        self.update_op = None
        self.data_exp = None
        self.data_var = None

    def get_param_num(self):
        return self.feats.num

    def get_params(self):
        return self.feats.values

    def set_params(self, value=None):
        if value is None:
            self.feats.values = np.random.uniform(-0.1, 0.1, size=self.get_param_num())
        else:
            self.feats.values = np.ones(self.get_param_num()) * value

    def get_order(self, level=None):
        return self.feats.get_order(level)

    def get_value(self, seq_list, depend_on=None):
        w = self.feats.seq_list_weight(seq_list, depend_on)
        return np.array(w)

    def get_gradient(self, data_list, data_scalar, sample_list, sample_scalar):

        # data_scalar = np.ones(len(data_list)) / len(data_list)
        # sample_len = np.array([len(x) for x in sample_list])
        # sample_scalar = self.len_factor[sample_len - self.config.min_len] / len(sample_list)

        exp_d = self.feats.seq_list_count(data_list, data_scalar) if self.data_exp is None else self.data_exp
        exp_s = self.feats.seq_list_count(sample_list, sample_scalar)
        return exp_d - exp_s - self.config.L2_reg * self.get_params()

    def update(self, data_list, data_scalars, sample_list, sample_scalars, learning_rate=1.0):
        g = self.get_gradient(data_list, data_scalars, sample_list, sample_scalars)
        if self.data_var is not None:
            g /= self.data_var
        d = self.update_op.update(-g, learning_rate)
        self.feats.values += d

    def initialize(self):
        if self.feats.num == 0:
            self.feats.load_from_seqs(self.data_seq_list)
        else:
            print('[{}.{}] Features exist. Don\'t reload features'.format(__name__, self.__class__.__name__))

        self.update_op = wb.ArrayUpdate(self.get_param_num(), {'name': self.opt_method})

        if self.config.pre_compute_data_exp:
            with wb.processing('compute feat-exp on data'):
                scalar = np.ones(len(self.data_seq_list)) / len(self.data_seq_list)
                self.data_exp = self.feats.seq_list_count(self.data_seq_list, scalar)

        if self.config.pre_compute_data_var:
            with wb.processing('compute feat-var on data'):
                self.data_var = self.compute_data_var(self.data_seq_list)
                self.data_var = np.maximum(self.data_var, self.config.var_gap)

    def save(self, fname):
        with open(fname + '.feat', 'wt') as f:
            self.feats.save(f)

    def restore(self, fname):
        with open(fname + '.feat', 'rt') as f:
            self.feats.restore(f)

    def compute_data_var(self, seq_list):
        n = len(seq_list)
        scalar = np.ones(n) / n
        exp = self.feats.seq_list_count(seq_list, scalar)
        exp2 = self.feats.seq_list_count2(seq_list, scalar)
        return exp2 - exp ** 2 + self.config.L2_reg


class NetPhi(Base):
    def __init__(self, config, data, opt_method='sgd', device='/gpu:0'):
        self.config = config
        self.data = data

        # revise the opt_method
        config.opt_method = opt_method

        self.train_net = wodnet.NetBase(config, is_training=True, device=device, name='trfnet', reuse=None)
        self.eval_net = wodnet.NetBase(config, is_training=False, device=device, name='trfnet', reuse=True)

        self.saver = tf.train.Saver(self.train_net.vars)

    def get_param_num(self):
        return self.train_net.get_param_num()

    def get_value(self, seq_list, depend_on=None):
        return self.eval_net.get_phi(seq.get_x(seq_list))

    def get_value_for_train(self, seq_list):
        return self.train_net.get_phi(seq.get_x(seq_list))

    def initialize(self):
        pass

    def update(self, data_list, data_scalars, sample_list, sample_scalars, learning_rate=1.0):
        self.train_net.update(seq.get_x(data_list), data_scalars,
                              seq.get_x(sample_list), sample_scalars,
                              learning_rate=learning_rate)

    def save(self, fname):
        self.saver.save(tf.get_default_session(), fname + '.ckpt')

    def restore(self, fname):
        self.saver.restore(tf.get_default_session(), fname + '.ckpt')
