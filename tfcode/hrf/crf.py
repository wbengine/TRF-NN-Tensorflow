import tensorflow as tf
import os
import json
import time
from copy import deepcopy
from collections import OrderedDict
from base import *
from lm import *

from trf.common import feat2 as feat
from . import pot, tagphi, mixphi, alg


class Config(wb.Config):
    def __init__(self, data):
        Config.value_encoding_map[lr.LearningRate] = str

        # self.min_len = data.get_min_len()
        # self.max_len = data.get_max_len()
        # self.pi_true = data.get_pi_true()
        # self.pi_0 = data.get_pi0(self.pi_true)
        self.word_vocab_size = data.get_vocab_size()
        self.tag_vocab_size = data.get_tag_size()
        self.beg_tokens = data.get_beg_tokens()  # [word_beg_token, tag_beg_token]
        self.end_tokens = data.get_end_tokens()  # [word_end_token, tag_end_token]

        # potential features
        #  for tags
        self.tag_config = tagphi.TagConfig(data)  # discrete features
        self.mix_config = mixphi.MixNetConfig(data)  # discrete features

        # for words
        # self.word_feat_config = None  # pot.FeatConfig(), discrete features
        # self.word_net_config = None   # pot.NetConfig(), nerual networks

        # init zeta
        # self.init_logz = self.get_initial_logz()

        # AugSA
        self.train_batch_size = 100
        # self.sample_batch_size = 100
        # self.chain_num = 10
        # self.multiple_trial = 1
        # self.sample_sub = 1
        # self.jump_width = 1
        # self.auxiliary_type = 'lstm'
        # self.auxiliary_config = lstmlm.Config(data)

        # learning rate
        self.lr_tag = lr.LearningRateEpochDelay(1.0)
        self.lr_mix = lr.LearningRateEpochDelay(1.0)
        # self.lr_logz = lr.LearningRateEpochDelay(1.0)
        self.opt_tag = 'sgd'
        self.opt_mix = 'sgd'
        # self.opt_logz_method = 'sgd'
        # self.zeta_gap = 10
        self.max_epoch = 100

        # dbg
        self.write_dbg = False

    def create_feat_config(self, feat_files, L2_reg=1e-6):
        feat_dict_all = feat.read_feattype_file(feat_files)

        tag_feat = dict()
        mix_feat = dict()
        wod_feat = dict()

        for ftype, cutoff in feat_dict_all.items():
            if ftype.find('w') != -1 and ftype.find('c') != -1:
                mix_feat[ftype] = cutoff
            elif ftype.find('w') != -1:
                wod_feat[ftype] = cutoff
            elif ftype.find('c') != -1:
                tag_feat[ftype] = cutoff

        self.tag_config.feat_dict = tag_feat
        self.tag_config.L2_reg = L2_reg
        self.mix_config.feat_dict = mix_feat
        self.mix_config.L2_reg = L2_reg
        # if len(wod_feat) > 0:
        #     self.word_feat_config = pot.FeatConfig()
        #     self.word_feat_config.feat_dict = wod_feat
        #     self.word_feat_config.L2_reg = L2_reg
        # else:
        #     self.word_feat_config = None

    # def get_initial_logz(self, c=None):
    #     if c is None:
    #         c = np.log(self.word_vocab_size) + np.log(self.tag_vocab_size)
    #     len_num = self.max_len - self.min_len + 1
    #     logz = c * (np.linspace(1, len_num, len_num))
    #     return logz

    def __str__(self):
        s = 'crf'

        if self.tag_config is not None:
            s += '_' + str(self.tag_config)
        if self.mix_config is not None:
            s += '_' + str(self.mix_config)

        return s


class CRF(object):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):

        self.config = config
        self.data = data
        self.logdir = logdir
        self.name = name

        # phi for tags
        self.phi_tag = tagphi.TagPhi(config.tag_config, data.datas[0], config.opt_tag)
        self.phi_mix = mixphi.create(config.mix_config, data.datas[0], config.opt_mix, device)

        assert self.phi_tag.get_order() >= self.phi_mix.get_order()

        self.marginal_used_map = None

        # learning rate
        self.cur_lr_tag = 1.0
        self.cur_lr_mix = 1.0

        # training info
        self.training_info = {'trained_step': 0,
                              'trained_epoch': 0,
                              'trained_time': 0}

        # debuger
        self.write_files = wb.FileBank(os.path.join(logdir, name + '.dbg'))
        # time recorder
        self.time_recoder = wb.clock()
        # default save name
        self.default_save_name = os.path.join(self.logdir, self.name + '.mod')

    def save(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[CRF] save to', fname)
        with open(fname + '.config', 'wt') as f:
            json.dump(self.training_info, f, indent=4)
            f.write('\n')
            self.config.save(f)
        self.phi_tag.save(fname)
        self.phi_mix.save(fname)

    def restore(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[CRF] restore from', fname)
        with open(fname + '.config', 'rt') as f:
            self.training_info = wb.json_load(f)
            print(json.dumps(self.training_info, indent=2))
        self.phi_tag.restore(fname)
        self.phi_mix.restore(fname)

    def exist_model(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        return wb.exists(fname + '.config')

    @property
    def session(self):
        return tf.get_default_session()

    def phi(self, seq_list):
        """
        Args:
            seq_list: a list of Seq()

        Returns:

        """
        a = np.zeros(len(seq_list))
        if self.phi_tag is not None:
            a += self.phi_tag.get_value(seq_list)
        if self.phi_mix is not None:
            a += self.phi_mix.get_value(seq_list)
        return a

    def phi_depend_on_tag(self, seq_list, depend_on_pos):
        """
        compute the phi only depending on the tags at given position
        Args:
            seq_list: a list of Seq()
            depend_on_pos: integer

        Returns:
            the phis
        """
        a = np.zeros(len(seq_list))
        if self.phi_tag is not None:
            a += self.phi_tag.get_value(seq_list, depend_on=(1, depend_on_pos))
        if self.phi_mix is not None:
            a += self.phi_mix.get_value(seq_list, depend_on=(1, depend_on_pos))

        return a

    def get_tag_logps(self, seq_list, tag_pos):
        logps = self.phi_tag.get_propose_logps(seq_list, tag_pos) + self.phi_mix.get_propose_logps(seq_list, tag_pos)
        logps -= logsumexp(logps, axis=-1, keepdims=True)
        return logps

    def logps(self, seq_list):
        """
        compute the logprobs of a list of Seq()
        Args:
            seq_list: a list of Seq()
        Returns:
            a np.arrdy()
        """
        logp = self.phi(seq_list) - self.logz([s.x[0] for s in seq_list])

        return logp

    def get_log_probs(self, seq_list, is_norm=True, batch_size=100):

        logps = np.zeros(len(seq_list))

        for i in range(0, len(seq_list), batch_size):
            if is_norm:
                logps[i: i+batch_size] = self.logps(seq_list[i: i+batch_size])
            else:
                logps[i: i+batch_size] = self.phi(seq_list[i: i+batch_size])

        return logps

    def get_logzs(self, x_list, batch_size=100):
        """separated into batch to accelerate the computation and limint the memory size"""
        n = len(x_list)

        logpx = np.zeros(n)
        for i in range(0, n, batch_size):
            logpx[i: i+batch_size] = self.logz(x_list[i: i+batch_size])
        return logpx

    def seq_beg_end_idxs(self, order):
        if order == 2:
            seq_beg_idxs = [self.config.beg_tokens[1]]
            seq_end_idxs = [self.config.end_tokens[1]]
        else:
            seq_beg_idxs = [sp.map_list(x, self.config.tag_vocab_size) for x in
                            sp.SeqIter(order-1, self.config.tag_vocab_size, beg_token=self.config.beg_tokens[1])]
            seq_end_idxs = [sp.map_list(x, self.config.tag_vocab_size) for x in
                            sp.SeqIter(order-1, self.config.tag_vocab_size, end_token=self.config.end_tokens[1])]

        return seq_beg_idxs, seq_end_idxs

    def logz(self, x_list):
        """given the word sequence, compute the normalization constants"""
        order = self.phi_tag.get_order()
        trans_mat, trans_mat_last = self.phi_tag.get_trans_matrix()
        emiss_mat_list = self.phi_mix.get_emission_vectors(order, x_list)
        seq_beg_idxs, seq_end_idxs = self.seq_beg_end_idxs(order)

        logz_list = []
        for x, emiss_vecs in zip(x_list, emiss_mat_list):

            fb = alg.ForwardBackward(trans_mat, trans_mat_last, emiss_vecs,
                                     beg_idxs=seq_beg_idxs, end_idxs=seq_end_idxs)

            logz = fb.logsum()
            logz_list.append(logz)

        logpx_all = np.array(logz_list)

        return logpx_all

    def get_tag(self, x_list):
        """perform decoding to get the optimal tags"""
        order = self.phi_tag.get_order()
        trans_mat, trans_mat_last = self.phi_tag.get_trans_matrix()
        emiss_mat_list = self.phi_mix.get_emission_vectors(order, x_list)
        seq_beg_idxs, seq_end_idxs = self.seq_beg_end_idxs(order)

        logp_list = []
        tags_list = []
        for x, emiss_vecs in zip(x_list, emiss_mat_list):
            assert order <= len(x)

            fp = alg.ForwardBackward(trans_mat, trans_mat_last, emiss_vecs,
                                     beg_idxs=seq_beg_idxs, end_idxs=seq_end_idxs)

            opt_tag_map, opt_logp = fp.decode()
            opt_tags = [sp.unfold_list(i, self.config.tag_vocab_size, order-1) for i in opt_tag_map]

            final_tags = [t[0] for t in opt_tags[0:-1]] + opt_tags[-1]

            # s = seq.Seq(np.stack([x, final_tags]))
            # print(self.phi([s])[0], opt_logp)

            tags_list.append(final_tags)
            logp_list.append(opt_logp)

        logp_list = np.array(logp_list)

        return tags_list, logp_list

    def marginal_logps(self, x_list):
        """

        Args:
            x_list: a list of word sequences

        Returns:
            2d array of shape (seq_len-order+1, self.config.tag_vocab_size ** order])
        """
        order = self.phi_tag.get_order()
        trans_mat, trans_mat_last = self.phi_tag.get_trans_matrix()
        emiss_mat_list = self.phi_mix.get_emission_vectors(order, x_list)
        seq_beg_idxs, seq_end_idxs = self.seq_beg_end_idxs(order)

        if self.marginal_used_map is None:
            idx1 = np.zeros(self.config.tag_vocab_size ** order, dtype='int32')
            idx2 = np.zeros(self.config.tag_vocab_size ** order, dtype='int32')
            for tags in sp.VecIter(order, self.config.tag_vocab_size):
                i = sp.map_list(tags[0:-1], self.config.tag_vocab_size)
                j = sp.map_list(tags[1:], self.config.tag_vocab_size)
                k = sp.map_list(tags, self.config.tag_vocab_size)
                idx1[k] = i
                idx2[k] = j
            self.marginal_used_map = (idx1, idx2)

        logps_list = []
        logz_list = []
        for emiss_mat in emiss_mat_list:

            fp = alg.ForwardBackward(trans_mat, trans_mat_last, emiss_mat,
                                     beg_idxs=seq_beg_idxs,
                                     end_idxs=seq_end_idxs)

            logps = fp.logps()

            # 2-d matrix to 1-d vector
            logps_legal = logps[:, self.marginal_used_map[0], self.marginal_used_map[1]]

            # normalize
            logz = fp.logsum()
            logps_legal -= logz

            # print(logz)
            # print(logsumexp(logps_legal, axis=tuple(range(1, order+1))))
            # print(np.exp(logsumexp(logps_legal, axis=tuple(range(1, order+1)))))

            logps_list.append(logps_legal)
            logz_list.append(logz)

        return logps_list, logz_list

    def eval(self, seq_list):
        logps = self.get_log_probs(seq_list)
        nll = -np.mean(logps)
        words = np.sum([len(x)-1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)

        return nll, ppl

    def initialize(self):
        # create features
        print('[CRF] init tag features.')
        self.phi_tag.initialize()
        print('[CRF] init mix features.')
        self.phi_mix.initialize()

        print('[CRF] tag_param_num = {:,}'.format(self.phi_tag.get_param_num()))
        print('[CRF] mix_param_num = {:,}'.format(self.phi_mix.get_param_num()))

    def update(self, data_list):
        # compute the scalars
        data_scalar = np.ones(len(data_list)) / len(data_list)

        with self.time_recoder.recode('update_fp'):
            logps_list, _ = self.marginal_logps([s.x[0] for s in data_list])

        # update phi
        with self.time_recoder.recode('update_tag'):
            self.phi_tag.update(data_list, data_scalar, data_list, data_scalar,
                                sample_fp_logps_list=logps_list,
                                learning_rate=self.cur_lr_tag)

        with self.time_recoder.recode('update_mix'):
            self.phi_mix.update(data_list, data_scalar, data_list, data_scalar,
                                sample_fp_logps_list=logps_list,
                                learning_rate=self.cur_lr_mix)

        return None

    def train(self, print_per_epoch=0.1, operation=None):

        # initialize
        self.initialize()

        if self.exist_model():
            self.restore()

        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]

        print('[CRF] [Train]...')
        print('train_list=', len(train_list))
        print('valid_list=', len(valid_list))
        time_beginning = time.time()
        model_train_nll = []

        self.data.train_batch_size = self.config.train_batch_size
        self.data.is_shuffle = True
        last_epoch = 0
        print_next_epoch = 0
        for step, data_seqs in enumerate(self.data):
            epoch = self.data.get_cur_epoch()

            # update training information
            self.training_info['trained_step'] += 1
            self.training_info['trained_epoch'] = self.data.get_cur_epoch()
            self.training_info['trained_time'] = (time.time() - time_beginning) / 60

            # update paramters
            with self.time_recoder.recode('update'):
                # learining rate
                self.cur_lr_tag = self.config.lr_tag.get_lr(step+1, epoch)
                self.cur_lr_mix = self.config.lr_mix.get_lr(step+1, epoch)
                self.cur_lr_tag = self.config.lr_tag.get_lr(step + 1, epoch)
                self.cur_lr_mix = self.config.lr_mix.get_lr(step + 1, epoch)
                # update
                self.update(data_seqs)

            # evaulate the NLL
            with self.time_recoder.recode('eval_train_nll'):
                model_train_nll.append(self.eval(data_seqs)[0])

            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                time_since_beg = (time.time() - time_beginning) / 60

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                info['lr_tag'] = '{:.2e}'.format(self.cur_lr_tag)
                info['lr_mix'] = '{:.2e}'.format(self.cur_lr_mix)
                info['train'] = np.mean(model_train_nll[-100:])
                info['valid'] = model_valid_nll
                log.print_line(info)

                print('[end]')

                # write time
                f = self.write_files.get('time')
                f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, time_since_beg))
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.items()]) + '\n')
                f.flush()

            if epoch >= self.config.max_epoch:
                print('[CRF] train stop!')
                self.save()
                if operation is not None:
                    operation.perform(step, epoch)
                break

            if int(epoch) > last_epoch:
                self.save()
                last_epoch = int(self.data.get_cur_epoch())

            ###########################
            # extra operations
            ###########################
            if operation is not None:
                operation.run(step, epoch)

    def debug_get_logpx(self):
        self.initialize()
        self.save()

        print('tag_order=', self.phi_tag.get_order())

        for set_value in [1]:
            # init the values
            self.phi_tag.feats.values[:] = np.random.uniform(size=self.phi_tag.get_param_num()) * set_value
            self.phi_tag.need_update_trans_matrix = True

            self.phi_mix.set_params(set_value)

            x_list = [sp.random_seq(3, 4,
                                    self.config.word_vocab_size,
                                    self.config.beg_tokens[0],
                                    self.config.end_tokens[0]) for _ in range(5)]

            s_list = [seq.Seq(x) for x in x_list]

            logz1 = self.logz(x_list)

            logz2 = []
            p_sum = []
            for s in s_list:
                temp_phi = []
                temp_logp = []
                tag_iter = sp.SeqIter(len(s),
                                      self.config.tag_vocab_size,
                                      self.config.beg_tokens[1],
                                      self.config.end_tokens[1])
                for tags in tag_iter:
                    s.x[1] = tags
                    temp_phi.append(self.phi([s])[0])
                    temp_logp.append(self.logps([s])[0])

                logz2.append(logsumexp(temp_phi))
                p_sum.append(np.exp(logsumexp(temp_logp)))
            print('v={}, len={}, logz1={}, logz2={}, psum={}'.format(set_value,
                                                                     [len(x) for x in x_list],
                                                                     logz1, logz2, p_sum))

    def debug_get_tag(self):
        self.initialize()
        self.save()

        print('tag_order=', self.phi_tag.get_order())

        for set_value in [1]:
            # init the values
            np.random.seed(0)
            self.phi_tag.feats.values = np.random.uniform(size=self.phi_tag.get_param_num())
            self.phi_tag.need_update_trans_matrix = True

            self.phi_mix.set_params(set_value)

            for seq_len in [3, 4]:

                x_list = [sp.random_seq(seq_len, seq_len,
                                        self.config.word_vocab_size,
                                        self.config.beg_tokens[0],
                                        self.config.end_tokens[0]) for _ in range(5)]

                print('x_list=', x_list)

                s_list = [seq.Seq(x) for x in x_list]

                tags_list1, logp_list1 = self.get_tag(x_list)

                tags_list2 = []
                logp_list2 = []
                tag_iter = sp.SeqIter(seq_len,
                                      self.config.tag_vocab_size,
                                      self.config.beg_tokens[1],
                                      self.config.end_tokens[1])
                for tags in tag_iter:
                    for s in s_list:
                        s.x[1] = tags
                    logp = self.get_log_probs(s_list, is_norm=False)

                    if len(logp_list2) == 0:
                        logp_list2 = logp
                        tags_list2 = [s.x[1] for s in s_list]
                        continue

                    for i in range(len(s_list)):
                        if logp[i] > logp_list2[i]:
                            logp_list2[i] = logp[i]
                            tags_list2[i] = s_list[i].x[1].tolist()

                print('v={}, len={}'.format(set_value, seq_len))
                print(' tag={}  logp={}'.format(tags_list1, logp_list1))
                print(' tag={}  logp={}'.format(tags_list2, logp_list2))

    def debug_tag_logps(self):
        seq_list = [seq.Seq().random(3, 4,
                                     [self.config.word_vocab_size, self.config.tag_vocab_size],
                                     self.config.beg_tokens,
                                     self.config.end_tokens) for _ in range(5)]

        pos = [len(s)-2 for s in seq_list]
        logps1 = self.get_tag_logps(seq_list, pos)

        seqs_enumerate = seq.seq_list_enumerate_tag(seq_list, self.config.tag_vocab_size, pos)
        logps2 = self.get_log_probs(seqs_enumerate, is_norm=False)
        logps2 = np.reshape(logps2, (len(seq_list), -1))
        logps2 -= logsumexp(logps2, axis=-1, keepdims=True)

        print('len=', [len(s) for s in seq_list])
        print('pos=', pos)
        print('logps1=', logps1)
        print('logps2=', logps2)


    def debug_compute_grad(self):
        self.initialize()
        self.save()

        print('tag_order=', self.phi_tag.get_order())

        self.phi_tag.feats.values = np.random.uniform(size=self.phi_tag.get_param_num())
        self.phi_tag.need_update_trans_matrix = True
        self.phi_mix.set_params()

        for seq_len in [3, 4]:
            seq_list = [seq.Seq().random(seq_len, seq_len,
                                         [self.config.word_vocab_size, self.config.tag_vocab_size],
                                         self.config.beg_tokens,
                                         self.config.end_tokens)
                        for _ in range(4)]

            data_scalar = np.ones(len(seq_list)) / len(seq_list)
            logps_list = self.marginal_logps([s.x[0] for s in seq_list])

            g1 = [0, 0]
            g1[0] = self.phi_mix.get_exp(seq_list, data_scalar, logps_list)
            g1[1] = self.phi_tag.get_exp(seq_list, data_scalar, logps_list)

            sample_list = []
            sample_scalar = []
            for s, scalar in zip(seq_list, data_scalar):
                for tags in sp.SeqIter(len(s), self.config.tag_vocab_size, self.config.beg_tokens[1], self.config.end_tokens[1]):
                    s.x[1] = tags
                    sample_list.append(s.copy())
                    sample_scalar.append(scalar * np.exp(self.logps([s])[0]))

            print('sum_sample_scalar=', np.sum(sample_scalar))
            g2 = [0, 0]
            g2[0] = self.phi_mix.feats.seq_list_count(sample_list, sample_scalar)
            g2[1] = self.phi_tag.feats.seq_list_count(sample_list, sample_scalar)

            for g1, g2 in zip(g1, g2):
                print('len={} g1={}, g2={}, max_g1-g2={}'.format(seq_len, np.sum(g1), np.sum(g2), np.sum(g1-g2)))


class DefaultOps(wb.Operation):
    def __init__(self, m, test_seq_list):
        super().__init__()

        self.m = m
        self.test_seq_list = test_seq_list
        self.wod_seq_list = seq.get_x(test_seq_list)
        self.tag_seq_list = seq.get_h(test_seq_list)

        self.perform_next_epoch = 1.0
        self.perform_per_epoch = 1.0
        self.write_models = wb.mkdir(os.path.join(self.m.logdir, 'crf_models'))

    def perform(self, step, epoch):
        print('[Ops] performing')
        # tagging
        time_beg = time.time()
        tag_res_list, _ = self.m.get_tag(self.wod_seq_list)
        tag_time = time.time() - time_beg

        # write
        log.write_seq_to_file(tag_res_list, os.path.join(self.m.logdir, 'test_res.tag'))
        log.write_seq_to_file(self.tag_seq_list, os.path.join(self.m.logdir, 'test_correct.tag'))

        # compute wer
        P, R, F = seq.tag_error(self.tag_seq_list, tag_res_list)

        print('epoch={:.2f} P={:.2f} R={:.2f} F={:.2f} rescore_time={:.2f}'.format(
               epoch, P, R, F, tag_time / 60))

        res = wb.FRes(os.path.join(self.m.logdir, 'results_tag_err.log'))
        res_name = 'epoch%.2f' % epoch
        res.Add(res_name, ['P', 'R', 'F'], [P, R, F])


