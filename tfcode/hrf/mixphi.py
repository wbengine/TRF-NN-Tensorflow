import tensorflow as tf

from base import *
from trf.common import feat2 as feat
from . import densefeat, alg, mixnet
from .alg import logab_int


class MixFeatConfig(wb.Config):
    def __init__(self, data):
        self.feat_dict = {'w[1]c[1]': 0, 'c[1]w[1]': 0}
        self.L2_reg = 0
        self.tag_size = data.get_tag_size()

    def __str__(self):
        max_order = feat.Feats(self.feat_dict).get_order()
        return 'mix{}g'.format(max_order)


class MixNetConfig(mixnet.Config):
    def __init__(self, data):
        self.word_vocab = data.get_vocab_size()
        self.tag_size = data.get_tag_size()
        self.order = 1  # only support order=1

        super().__init__(self.word_vocab, self.tag_size ** self.order)

    def __str__(self):
        return 'mixnet{}g'.format(self.order)


def create(mix_config, data_seq_list, opt_method, device='/gpu:0'):
    if isinstance(mix_config, MixFeatConfig):
        return MixFeatPhi(mix_config, data_seq_list, opt_method)
    elif isinstance(mix_config, MixNetConfig):
        return MixNetPhi(mix_config, data_seq_list, opt_method, device)
    else:
        raise TypeError('[{}] create: not defined the config type={}'.format(__name__, type(mix_config)))


def seq_list_package(seq_list, pad_value=0):
    lengths = [len(s) for s in seq_list]
    inputs = np.ones([len(seq_list), np.max(lengths)]) * pad_value
    labels = np.ones_like(inputs) * pad_value

    for i, s in enumerate(seq_list):
        n = len(s)
        inputs[i][0:n] = s.x[0]
        labels[i][0:n] = s.x[1]

    return inputs, labels, lengths


def seq_list_unfold(inputs, labels, lengths):
    seq_list = []
    for x, y, n in zip(inputs, labels, lengths):
        seq_list.append(seq.Seq([x[0:n], y[0:n]]))
    return seq_list


class MixBase(object):
    def get_param_num(self):
        return 0

    def set_params(self, value=None):
        pass

    def get_value(self, seq_list, depend_on=None):
        return np.zeros(len(seq_list))

    def get_order(self):
        return 0

    def get_propose_logps(self, seq_list, tag_pos):
        return np.zeros((len(seq_list), self.config.tag_size))

    def initialize(self):
        pass

    def update(self, data_list, data_scalars, sample_list, sample_scalars, learning_rate=1.0, sample_fp_logps_list=None):
        pass

    def save(self, fname):
        pass

    def restore(self, fname):
        pass

    def get_emission_vectors(self, order, x_list):
        return 0


class MixFeatPhi(MixBase):
    def __init__(self, config, data_seq_list, opt_method):
        self.config = config  # MixFeatConfig()
        self.data_seq_list = data_seq_list
        self.opt_method = opt_method

        # mixture features
        self.feats = densefeat.Feats(self.config.feat_dict, dense_level=1, dense_vocab_size=self.config.tag_size)

        # update
        self.update_op = None

        self.time_recoder = wb.clock()

    def get_param_num(self):
        return self.feats.num

    def set_params(self, value=None):
        if value is None:
            self.feats.values = np.random.uniform(-0.1, 0.1, size=self.get_param_num())
        else:
            self.feats.values = np.ones(self.get_param_num()) * value

    def get_order(self):
        return self.feats.get_order()

    def get_propose_logps(self, seq_list, tag_pos):
        if isinstance(tag_pos, int):
            tag_pos = [tag_pos] * len(seq_list)

        logps = []
        for s, pos in zip(seq_list, tag_pos):
            temp_seqs = seq.seq_list_enumerate_tag([s], self.config.tag_size, pos)
            m = self.get_value(temp_seqs, depend_on=(1, pos))
            logps.append(m)

        return np.array(logps)

    def get_value(self, seq_list, depend_on=None):
        w = self.feats.seq_list_weight(seq_list, depend_on)
        return np.array(w)

    def get_gradient(self, data_list, data_scalar, sample_list, sample_scalar):
        exp_d = self.feats.seq_list_count(data_list, data_scalar)
        exp_s = self.feats.seq_list_count(sample_list, sample_scalar)
        return exp_d - exp_s - self.config.L2_reg * self.feats.values

    def get_exp(self, data_list, data_scalar, logps_list):

        if logps_list is None:
            return self.feats.seq_list_count(data_list, data_scalar)

        exp_s = np.zeros_like(self.feats.values)

        for s, scalar, src_logps in zip(data_list, data_scalar, logps_list):

            order = self.get_order()
            logps = alg.marginal_logps(src_logps, self.config.tag_size, order)


            for i, p in enumerate(logps):
                with self.time_recoder.recode('get_id'):
                    ids = self.feats.ngram_enumerate_ids(s, i, order)
                    if i == len(logps) - 1:
                        for j in range(1, order):
                            a = self.feats.ngram_enumerate_ids(s, i+j, order-j)
                            a = np.repeat(a, self.config.tag_size ** j, axis=0)
                            ids = np.concatenate([ids, a], axis=-1)
                with self.time_recoder.recode('add_prob'):
                    # for tags in range(ids.shape[0]):
                    #     for k in range(ids.shape[1]):
                    #         exp_s[ids[tags, k]] += probs[i][tags] * scalar

                    for id_k in np.transpose(ids):
                        for k, add_p in zip(id_k, np.exp(p) * scalar):
                            exp_s[k] += add_p

                    # for ids_for_tags, p_for_tags in zip(ids, p * scalar):
                    #     for k in ids_for_tags:
                    #         exp_s[k] += p_for_tags

        return exp_s

    # def get_gradient_fb(self, data_list, data_scalar, sample_list, sample_scalar, logps_list):
    #     exp_d = self.feats.seq_list_count(data_list, data_scalar)
    #     exp_s = self.get_exp(sample_list, sample_scalar, logps_list)
    #     return exp_d - exp_s - self.config.L2_reg * self.feats.values

    def update(self, data_list, data_scalars, sample_list, sample_scalars, learning_rate=1.0,
               data_fp_logps_list=None,
               sample_fp_logps_list=None):

        exp_d = self.get_exp(data_list, data_scalars, data_fp_logps_list)
        exp_s = self.get_exp(sample_list, sample_scalars, sample_fp_logps_list)
        g = exp_d - exp_s - self.config.L2_reg * self.feats.values

        # if sample_fp_logps_list is not None:
        #     # exact gradient
        #     g = self.get_gradient_fb(data_list, data_scalars, sample_list, sample_scalars, sample_fp_logps_list)
        # else:
        #     g = self.get_gradient(data_list, data_scalars, sample_list, sample_scalars)
        d = self.update_op.update(-g, learning_rate)
        self.feats.values += d

    def initialize(self):
        if self.feats.num == 0:
            self.feats.load_from_seqs(self.data_seq_list)
        else:
            print('[{}.{}] Features exist. Don\'t reload features'.format(__name__, self.__class__.__name__))

        self.update_op = wb.ArrayUpdate(self.get_param_num(), {'name': self.opt_method})

    def save(self, fname):
        with open(fname + '.mix.feat', 'wt') as f:
            self.feats.save(f)

    def restore(self, fname):
        with open(fname + '.mix.feat', 'rt') as f:
            self.feats.restore(f)

    def get_emission_vectors(self, order, x_list):
        """ return a list of emission matrix """
        return [self.get_one_emission_vectors(order, x) for x in x_list]

    def get_one_emission_vectors(self, order, x):
        """ return the log-probs"""
        if self.feats.get_order() > order-1:
            raise TypeError('[{}.{}] the order of mix-feat={} is larger than the given order-1={}'.format(
                                __name__, self.__class__.__name__, self.feats.get_order(), order-1
                            ))

        num = len(x) - order + 2
        dim = self.config.tag_size ** (order - 1)

        vecs = np.zeros((num, dim))
        # old_vecs = np.zeros((num, dim))

        s = seq.Seq(x, level=2)
        for i in range(num):
            vecs[i] = self.feats.ngram_enumerate(s, i, order-1)
            if i == num - 1:
                for j in range(1, order-1):
                    v = self.feats.ngram_enumerate(s, i+j, order-1-j)
                    vecs[i] += np.tile(v, self.config.tag_size ** j)

        # vecs = np.exp(vecs)
        return vecs


class MixNetPhi(MixBase):
    def __init__(self, config, data_seq_list, opt_method, device='/gpu:0'):
        self.config = config

        self.net = mixnet.Net(config, is_training=True, device=device, name='mixnet')
        self.saver = tf.train.Saver(self.net.vars)

    @property
    def session(self):
        return tf.get_default_session()

    def get_param_num(self):
        return self.net.run_parameter_num(self.session)

    def get_order(self):
        return self.config.order

    def get_value(self, seq_list, depend_on=None):
        inputs, labels, lengths = seq_list_package(seq_list)
        return self.net.run_phi(self.session, inputs, labels, lengths)

    def get_propose_logps(self, seq_list, tag_pos):
        assert self.get_order() == 1

        if isinstance(tag_pos, int):
            tag_pos = [tag_pos] * len(seq_list)

        logps_list = self.get_emission_vectors(2, [s.x[0] for s in seq_list])
        m = [logps[pos] for logps, pos in zip(logps_list, tag_pos)]
        return np.array(m)

    def save(self, fname):
        self.saver.save(self.session, fname + '.mix.ckpt')

    def restore(self, fname):
        self.saver.restore(self.session, fname + '.mix.ckpt')

    def initialize(self):
        pass

    def get_gradient_output(self, seq_list, seq_scalar, logps_list=None):
        if logps_list is not None:
            logps_list = alg.marginal_logps_list(logps_list, self.config.tag_size, self.get_order())
            logps_array = alg.logps_list_package(logps_list)
            probs_array = np.exp(logps_array)
        else:
            # compute one-hot
            max_len = np.max([len(s) for s in seq_list])
            dim = self.config.tag_size ** self.get_order()
            probs_array = np.zeros([len(seq_list), max_len, dim])
            for k, s in enumerate(seq_list):
                for i in range(0, len(s)-self.get_order()+1):
                    j = sp.map_list(s.x[1][i: i+self.get_order()], self.config.tag_size)
                    probs_array[k, i, j] = 1

        probs_array = np.reshape(seq_scalar, [-1, 1, 1]) * probs_array
        return probs_array

    def update(self, data_list, data_scalars, sample_list, sample_scalars, learning_rate=1.0,
               data_fp_logps_list=None,
               sample_fp_logps_list=None):

        data_grads = self.get_gradient_output(data_list, data_scalars, data_fp_logps_list)
        samp_grads = - self.get_gradient_output(sample_list, sample_scalars, sample_fp_logps_list)

        inputs, _, lengths = seq_list_package(data_list + sample_list)
        max_len = np.max(lengths)

        data_grads = np.pad(data_grads, [[0, 0], [0, max_len-data_grads.shape[1]], [0, 0]], mode='constant')
        samp_grads = np.pad(samp_grads, [[0, 0], [0, max_len-samp_grads.shape[1]], [0, 0]], mode='constant')
        grads = np.concatenate([data_grads, samp_grads], axis=0)

        self.net.run_update(self.session, inputs, lengths, grads, learning_rate=learning_rate)

    def get_emission_vectors(self, order, x_list):
        if order-1 != self.get_order():
            raise TypeError('[{}.{}] get_emission_vectors, the needed order-1={} != the model order={}'.format(
                __name__, self.__class__.__name__, order-1, self.get_order()
            ))

        inputs, lengths = reader.produce_data_to_array(x_list)
        outputs = self.net.run_outputs(self.session, inputs, lengths)
        logps_list = alg.logps_list_unfold(outputs,
                                           lengths - self.get_order() + 1)
        return logps_list














