import tensorflow as tf
import time
import json
import os
from collections import OrderedDict

from base import *
from lm import *
from trf.common import *
from trf.sa import simulater, pot
from trf.nce.pot import NormLinear


class Config(wb.Config):
    def __init__(self, data):
        Config.value_encoding_map[lr.LearningRate] = str

        self.min_len = data.get_min_len()
        self.max_len = data.get_max_len()
        self.vocab_size = data.get_vocab_size()
        self.pi_true = data.get_pi_true()
        self.pi_0 = data.get_pi0(self.pi_true, add=0.01)
        self.beg_token = data.get_beg_token()
        self.end_token = data.get_end_token()

        self.word_average = True

        # prior model path
        self.prior_model_path = None

        # for discrete features
        self.feat_config = pot.FeatConfig(data)

        # for network features
        self.net_config = pot.NetConfig(data)

        # init zeta
        self.init_logz = self.get_initial_logz()

        # AugSA
        self.train_batch_size = 1000
        self.sample_batch_size = 100
        self.chain_num = 10
        self.multiple_trial = 10
        self.sample_sub = 10
        self.jump_width = 5
        self.auxiliary_config = lstmlm.Config(data)

        # learning rate
        self.lr_feat = lr.LearningRateEpochDelay(1.0)
        self.lr_net = lr.LearningRateEpochDelay(1.0)
        self.lr_logz = lr.LearningRateEpochDelay(1.0)
        self.opt_feat_method = 'sgd'
        self.opt_net_method = 'sgd'
        self.opt_logz_method = 'sgd'
        self.max_epoch = 100

        # dbg
        self.write_dbg = False

    def get_initial_logz(self, c=None):
        if c is None:
            c = np.log(self.vocab_size)
        len_num = self.max_len - self.min_len + 1
        logz = c * (np.linspace(1, len_num, len_num))
        return logz

    def __str__(self):
        s = 'trf_sa{}'.format(self.chain_num)
        if self.prior_model_path is not None:
            s += '_priorlm'

        if self.feat_config is not None:
            s += '_' + str(self.feat_config)
        if self.net_config is not None:
            s += '_' + str(self.net_config)

        # if self.data_sampler is not None:
        #     s += '_data{}'.format(self.data_sampler.split(':')[0])
        return s


class TRF(object):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):
        self.config = config
        self.data = data
        self.logdir = logdir
        self.name = name

        # prior LM, q(x)
        self.priorlm = priorlm.LSTMLM(config.prior_model_path, device=device) \
            if config.prior_model_path is not None else priorlm.EmptyLM()

        # phi
        self.phi_feat = pot.FeatPhi(config.feat_config, data, config.opt_feat_method) \
            if config.feat_config is not None else pot.Base()
        self.phi_net = pot.NetPhi(config.net_config, data, config.opt_net_method, device=device) \
            if config.net_config is not None else pot.Base()

        # logZ
        self.norm_const = pot.Norm(config, data, config.opt_logz_method)
        # self.norm_const = pot.NormFixed(config, data, config.opt_logz_method)

        # simulater
        self.simulater = simulater.SimulaterLSTM(self.config.auxiliary_config, device=device)
        # length jump probabilities
        self.gamma = sp.len_jump_distribution(self.config.min_len,
                                              self.config.max_len,
                                              self.config.jump_width)
        # sample sequences
        self.sample_seq = reader.produce_data_to_trf(
            [sp.random_seq(self.config.min_len,
                           self.config.max_len,
                           self.config.vocab_size,
                           beg_token=self.config.beg_token,
                           end_token=self.config.end_token,
                           pi=self.config.pi_true)
             for _ in range(self.config.chain_num)])
        # save the adjacent mcmc state
        # i.e. the adjacent mcmc sequences
        self.sample_mcmc = []

        # learning rate
        self.cur_lr_feat = 1.0
        self.cur_lr_net = 1.0
        self.cur_lr_logz = 1.0

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

        # debug variables
        self.lj_times = 1
        self.lj_success = 0
        self.lj_rate = 1
        self.mv_times = 1
        self.mv_success = 0
        self.mv_rate = 1
        self.sample_cur_pi = np.zeros(self.config.max_len+1)     # current pi
        self.sample_acc_count = np.zeros(self.config.max_len+1)  # accumulated count

    @property
    def global_step(self):
        return self.phi_net.global_step

    def save(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[TRF] save to', fname)
        with open(fname + '.config', 'wt') as f:
            json.dump(self.training_info, f, indent=4)
            f.write('\n')
            self.config.save(f)
        self.phi_feat.save(fname)
        self.phi_net.save(fname)
        self.norm_const.save(fname + '.norm')

    def restore(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[TRF] restore from', fname)
        with open(fname + '.config', 'rt') as f:
            self.training_info = wb.json_load(f)
            print(json.dumps(self.training_info, indent=2))
        self.phi_feat.restore(fname)
        self.phi_net.restore(fname)
        self.norm_const.restore(fname + '.norm')

    def restore_nce_model(self, fname):
        print('[TRF] restore nce model from', fname)
        self.phi_feat.restore(fname)
        self.phi_net.restore(fname)

        nce_norm = NormLinear(self.config, self.data)
        nce_norm.restore(fname + '.norm')
        logz = nce_norm.get_logz(np.linspace(self.config.min_len,
                                             self.config.max_len+1,
                                             self.config.max_len - self.config.min_len + 1))
        self.norm_const.set_logz(logz)
        self.norm_const.set_logz1(self.true_logz(self.config.min_len)[0])

    def exist_model(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        return wb.exists(fname + '.norm')

    def phi(self, input_x, input_n):
        seq_list = reader.extract_data_from_array(input_x, input_n)
        return self.phi_feat.get_value(seq_list) + self.phi_net.get_value(seq_list)

    def logps(self, input_x, input_n, for_eval=True):
        phi = self.phi(input_x, input_n)
        seq_list = reader.extract_data_from_array(input_x, input_n)

        if np.any(input_n < self.config.min_len) or np.any(input_n > self.config.max_len):
            raise TypeError('min_len={}, max_len={} lens={}'.format(min(input_n), max(input_n), input_n))

        if for_eval:
            # using pi_true
            logp_m = phi + self.priorlm.get_log_probs(seq_list) + \
                     np.log(self.config.pi_true[input_n]) - self.norm_const.get_logz(input_n)
        else:
            # using pi_0
            logp_m = phi + self.priorlm.get_log_probs(seq_list) + \
                     np.log(self.config.pi_0[input_n]) - self.norm_const.get_logz(input_n)

        return logp_m

    def get_log_probs(self, seq_list, is_norm=True, for_eval=True):
        seqs, indexs = self.data.cut_data_to_length(seq_list,
                                                    maxlen=self.config.max_len)

        logps = np.zeros(len(seqs))

        minibatch = self.config.chain_num * self.config.multiple_trial
        for i in range(0, len(seqs), minibatch):
            input_x, input_n = reader.produce_data_to_array(seqs[i: i+minibatch])
            if is_norm:
                logps[i: i+minibatch] = self.logps(input_x, input_n, for_eval)
            else:
                logps[i: i+minibatch] = self.phi(input_x, input_n)

        res = []
        for idx_b, idx_e in indexs:
            res.append(np.sum(logps[idx_b: idx_e]))

        return np.array(res)

    def eval(self, seq_list, for_eval=True):
        logps = self.get_log_probs(seq_list, for_eval=for_eval)
        nll = -np.mean(logps)
        words = np.sum([len(x)-1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)

        return nll, ppl

    def true_logz(self, max_len=None):
        if max_len is None:
            max_len = self.config.max_len

        logz = np.zeros(max_len - self.config.min_len + 1)
        for l in range(self.config.min_len, max_len+1):
            x_batch = [x for x in sp.SeqIter(l, self.config.vocab_size,
                                             beg_token=self.config.beg_token,
                                             end_token=self.config.end_token)]
            logz[l-self.config.min_len] = sp.log_sum(self.get_log_probs(x_batch, is_norm=False))
        return logz

    def local_jump_batch(self, input_x, input_n):

        batch_size = len(input_x)
        old_seqs = reader.extract_data_from_trf(input_x, input_n)
        new_seqs = [None] * batch_size
        acc_logps = np.zeros(batch_size)

        next_n = np.array([np.random.choice(self.config.max_len+1, p=self.gamma[n]) for n in input_n])
        jump_rate = np.array([np.log(self.gamma[j, k]) - np.log(self.gamma[k, j]) for j, k in zip(next_n, input_n)])

        inc_index = np.where(next_n > input_n)[0]
        des_index = np.where(next_n < input_n)[0]
        equ_index = np.where(next_n == input_n)[0]

        if len(equ_index) > 0:
            acc_logps[equ_index] = 0
            for i in equ_index:
                new_seqs[i] = old_seqs[i]

        if len(inc_index) > 0:
            chain_num = len(inc_index)
            init_seqs = input_x[inc_index]
            init_lens = input_n[inc_index]
            final_lens = next_n[inc_index]
            cur_jump_rate = jump_rate[inc_index]

            x_repeat = init_seqs.repeat(self.config.multiple_trial, axis=0)
            n_repeat = init_lens.repeat(self.config.multiple_trial, axis=0)
            m_repeat = final_lens.repeat(self.config.multiple_trial, axis=0)
            y_multiple, logp_y_multiple = self.simulater.local_jump_propose(x_repeat, n_repeat-1, m_repeat-1)
            # add end-tokens
            y_multiple = np.pad(y_multiple, [[0, 0], [0, 1]], 'constant')
            y_multiple[np.arange(y_multiple.shape[0]), m_repeat-1] = self.config.end_token
            logw_multiple_y = self.logps(y_multiple, m_repeat, for_eval=False)

            draw_idxs = [sp.log_sample(sp.log_normalize(x)) for x in logw_multiple_y.reshape((chain_num, -1))]
            draw_idxs_flatten = [i * self.config.multiple_trial + draw_idxs[i] for i in range(len(draw_idxs))]
            new_y = y_multiple[draw_idxs_flatten]
            new_m = m_repeat[draw_idxs_flatten]
            g_add = logp_y_multiple[draw_idxs_flatten]
            assert np.all(new_m == final_lens)

            cur_acc_logps = cur_jump_rate + \
                            logsumexp(logw_multiple_y.reshape((chain_num, -1)), axis=-1) - \
                            np.log(self.config.multiple_trial) - g_add - self.logps(init_seqs, init_lens, for_eval=False)

            # summary results
            acc_logps[inc_index] = cur_acc_logps
            for i, y in zip(inc_index, reader.extract_data_from_trf(new_y, new_m)):
                new_seqs[i] = y

        if len(des_index) > 0:
            chain_num = len(des_index)
            init_seqs = input_x[des_index]
            init_lens = input_n[des_index]
            final_lens = next_n[des_index]
            cur_jump_rate = jump_rate[des_index]

            y_repeat = init_seqs.repeat(self.config.multiple_trial, axis=0)
            n_repeat = init_lens.repeat(self.config.multiple_trial, axis=0)
            m_repeat = final_lens.repeat(self.config.multiple_trial, axis=0)
            x_multiple, logp_x_multiple = self.simulater.local_jump_propose(y_repeat, m_repeat-1, n_repeat-1)
            # add end-token
            x_multiple = np.pad(x_multiple, [[0, 0], [0, 1]], 'constant')
            x_multiple[np.arange(x_multiple.shape[0]), n_repeat-1] = self.config.end_token
            # set the initial_sequences
            for i in range(chain_num):
                # if len(x_multiple[i * self.config.multiple_trial]) != len(init_seqs[i]):
                #     print(x_multiple[i * self.config.multiple_trial])
                #     print(init_seqs[i])
                n = n_repeat[i * self.config.multiple_trial]
                x_multiple[i * self.config.multiple_trial, 0: n] = init_seqs[i, 0: init_lens[i]]
            logw_multiple_x = self.logps(x_multiple, n_repeat, for_eval=False)

            g_add = self.simulater.local_jump_condition(init_seqs, init_lens-1, final_lens-1)
            new_y = np.array(init_seqs)
            new_m = final_lens
            new_y[np.arange(new_y.shape[0]), new_m-1] = self.config.end_token

            cur_acc_logps = cur_jump_rate + \
                            np.log(self.config.multiple_trial) + g_add + self.logps(new_y, new_m, for_eval=False) - \
                            logsumexp(logw_multiple_x.reshape((chain_num, -1)), axis=-1)

            # summary results
            acc_logps[des_index] = cur_acc_logps
            for i, y in zip(des_index, reader.extract_data_from_trf(new_y, new_m)):
                new_seqs[i] = y

        res_seqs = []
        for i, (acc_logp, old_x, new_x) in enumerate(zip(acc_logps, old_seqs, new_seqs)):
            self.lj_times += 1
            try:
                if sp.accept_logp(acc_logp):
                    self.lj_success += 1
                    res_seqs.append(new_x)
                else:
                    res_seqs.append(old_x)
            except ValueError:
                print('acc_logps=', acc_logps)
                print('acc_logp=', acc_logp)
                print('type=', type(acc_logps))
                raise ValueError

            out_line = '[local jump] {}->{} acc_logp={:.2f} '.format(
                        len(old_x), len(new_x), float(acc_logp)
                        )
            out_line += 'acc_rate={:.2f}% '.format(100.0 * self.lj_success / self.lj_times)
            out_line += '[{}/{}] '.format(self.lj_success, self.lj_times)
            f = self.write_files.get('markov')
            f.write(out_line + '\n')
            f.flush()

        return reader.produce_data_to_trf(res_seqs)

    def markov_move_batch_sub(self, input_x, input_n, beg_pos, end_pos):
        """
        draw the values at positions x[beg_pos: beg_pos+sub_len]
        using multiple-trial MH sampling
        """
        chain_num = len(input_x)

        # propose multiple ys for each x in input_x
        x_repeat = input_x.repeat(self.config.multiple_trial, axis=0).astype(input_x.dtype)
        n_repeat = input_n.repeat(self.config.multiple_trial, axis=0).astype(input_n.dtype)

        # propose y
        multiple_y, logp_multiple_y = self.simulater.markov_move_propose(x_repeat, n_repeat-1,
                                                                         beg_pos, end_pos)
        # return logp_multiple_y
        multiple_y[np.arange(len(multiple_y)), n_repeat-1] = self.config.end_token  # set end tokens
        logw_multiple_y = self.phi(multiple_y, n_repeat) - logp_multiple_y
        # logw_multiple_y = self.phi(multiple_y, n_repeat) -\
        #                   self.simulater.markov_move_condition(self.get_session(),
        #                                                        x_repeat, n_repeat-1, multiple_y,
        #                                                        beg_pos, end_pos)

        # sample y
        draw_idxs = [sp.log_sample(sp.log_normalize(x)) for x in logw_multiple_y.reshape((chain_num, -1))]
        draw_idxs_flatten = [i * self.config.multiple_trial + draw_idxs[i] for i in range(len(draw_idxs))]
        new_y = multiple_y[draw_idxs_flatten]

        # draw reference x
        # as is independence sampling
        # there is no need to generate new samples
        logw_multiple_x = np.array(logw_multiple_y)
        logw_multiple_x[draw_idxs_flatten] = self.phi(input_x, input_n) - \
                                         self.simulater.markov_move_condition(input_x, input_n-1,
                                                                              beg_pos, end_pos)

        # compute the acceptance rate
        weight_y = logsumexp(logw_multiple_y.reshape((chain_num, -1)), axis=-1)
        weight_x = logsumexp(logw_multiple_x.reshape((chain_num, -1)), axis=-1)
        acc_logps = weight_y - weight_x

        # acceptance
        acc_probs = np.exp(np.minimum(0., acc_logps))
        accept = acc_probs >= np.random.uniform(size=acc_probs.shape)
        res_x = np.where(accept.reshape(-1, 1), new_y, input_x)
        self.mv_times += accept.size
        self.mv_success += accept.sum()

        out_line = '[Markov move] acc=' + log.to_str(acc_probs) + \
                   ' weight_y=' + log.to_str(weight_y) + \
                   ' weight_x=' + log.to_str(weight_x)
        f = self.write_files.get('markov')
        f.write(out_line + '\n')
        f.flush()

        return res_x, input_n

    def markov_move_batch(self, input_x, input_n):

        max_len = np.max(input_n)
        sub_sent = self.config.sample_sub if self.config.sample_sub > 0 else max_len

        # skip the beg / end tokens
        for beg_pos in range(1, max_len-1, sub_sent):
            # find all the sequence whose length is larger than beg_pos
            idx = np.where(input_n-1 > beg_pos)[0]
            # desicide the end position
            end_pos = min(max_len-1, beg_pos + sub_sent)
            local_x, local_n = self.markov_move_batch_sub(input_x[idx], input_n[idx], beg_pos, end_pos)

            input_x[idx] = local_x
            input_n[idx] = local_n

        return input_x, input_n

    def trans_dim_MH(self, input_x, input_n):

        chain_num = len(input_x)

        y_seqs = []
        for i in range(chain_num):
            y_seqs += self.noise_sampler.get()
        y_multiple, y_len_multiple = reader.produce_data_to_trf(y_seqs)

        logp_y_multiple = self.get_log_probs(y_seqs)
        logg_y_multiple = self.noise_sampler.noise_logps(y_seqs)
        logw_y_multiple = logp_y_multiple - logg_y_multiple

        draw_idxs = [sp.log_sample(sp.log_normalize(x)) for x in logw_y_multiple.reshape((chain_num, -1))]
        draw_idxs_flatten = [i * self.config.multiple_trial + draw_idxs[i] for i in range(len(draw_idxs))]
        new_y = y_multiple[draw_idxs_flatten]
        new_n = y_len_multiple[draw_idxs_flatten]

        logw_x_multiple = np.array(logw_y_multiple)
        x_seqs = reader.extract_data_from_trf(input_x, input_n)
        logw_x_multiple[draw_idxs_flatten] = self.get_log_probs(x_seqs) - self.noise_sampler.noise_logps(x_seqs)

        # compute the acceptance rate
        weight_y = logsumexp(logw_y_multiple.reshape((chain_num, -1)), axis=-1)
        weight_x = logsumexp(logw_x_multiple.reshape((chain_num, -1)), axis=-1)
        acc_logps = weight_y - weight_x

        # acceptance
        acc_probs = np.exp(acc_logps)
        accept = acc_probs >= np.random.uniform(size=acc_probs.shape)

        x_list = reader.extract_data_from_trf(input_x, input_n)
        y_list = reader.extract_data_from_trf(new_y, new_n)
        res_list = [y if acc else x for acc, x, y in zip(accept, x_list, y_list)]

        self.mv_times += accept.size
        self.mv_success += accept.sum()

        f = self.write_files.get('markov')
        f.write('[Markov move] acc=' + to_str(acc_probs) + '\n')
        f.write('   curr_n=' + to_str(input_n, '{}') + '\n')
        f.write('   next_n=' + to_str(new_n, '{}') + '\n')
        f.write('   weight_x=' + to_str(weight_x) + '\n')
        f.write('   weight_y=' + to_str(weight_y) + '\n')
        f.write('   logpw_x=' + to_str(logw_x_multiple[draw_idxs_flatten]) + '\n')
        f.write('   logpw_y=' + to_str(logw_y_multiple[draw_idxs_flatten]) + '\n')
        f.flush()

        return reader.produce_data_to_trf(res_list)

    def sample(self, input_x, input_n):
        with self.time_recoder.recode('local_jump'):
            input_x, input_n = self.local_jump_batch(input_x, input_n)

        with self.time_recoder.recode('markov_move'):
            input_x, input_n = self.markov_move_batch(input_x, input_n)

        return input_x, input_n

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
            seq_list += reader.extract_data_from_trf(*self.sample_seq)   # copy the sequence

        if self.lj_times > 0:
            self.lj_rate = 0.9 * self.lj_rate + 0.1 * (self.lj_success / self.lj_times)
        if self.mv_times > 0:
            self.mv_rate = 0.9 * self.mv_rate + 0.1 * (self.mv_success / self.mv_times)

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

    def test_sample(self):
        """
        Test the sample method
        """
        self.norm_const.set_logz(self.true_logz())
        print(self.norm_const.get_logz())

        # compute the marginal probability
        time_beg = time.time()
        comp_len = 3
        comp_pos = 1
        x_batch = [x for x in sp.SeqIter(comp_len, self.config.vocab_size, self.config.beg_token, self.config.end_token)]
        x_logp = self.get_log_probs(x_batch, True)
        print('x_logp', x_logp)
        local_p = np.zeros(self.config.vocab_size)
        for x, logp in zip(x_batch, x_logp):
            local_p[x[comp_pos]] += np.exp(logp)
        sample_local_p = np.zeros(self.config.vocab_size)

        # summary the length distribution
        sample_pi_all = np.zeros(self.config.max_len+1)
        sample_pi_cur = np.zeros(self.config.max_len+1)

        print('begin sampling:')
        batch_num = 100
        batch_size = 100
        for t in range(batch_num):
            sample_pi_cur.fill(0)

            sample_list = self.draw(batch_size)

            for s in sample_list:
                sample_pi_cur[len(s)] += 1

                if len(s) == comp_len:
                    sample_local_p[s[comp_pos]] += 1

            sample_pi_all += sample_pi_cur

            f = self.write_files.get('sample')
            for s in sample_list:
                f.write(' '.join([str(w) for w in s]) + '\n')
            f.flush()

            print('sample_pi=', sample_pi_all[self.config.min_len:] / np.sum(sample_pi_all))
            print('true_pi_0=', self.config.pi_0[self.config.min_len:])
            print('sample_p=', sample_local_p / np.sum(sample_pi_all))
            print('trup_p  =', local_p)
            time_since_beg = time.time() - time_beg
            print('t={} time={:.2f} time_per_step={:.2f}'.format(t, time_since_beg, time_since_beg/(t+1)))

    def update(self, data_list, sample_list):

        # compute the scalars
        data_scalar = np.ones(len(data_list)) / len(data_list)
        sample_len = np.array([len(x) for x in sample_list])
        sample_facter = np.array(self.config.pi_true[self.config.min_len:]) / \
                        np.array(self.config.pi_0[self.config.min_len:])
        sample_scalar = sample_facter[sample_len - self.config.min_len] / len(sample_list)

        # update feat-phi
        with self.time_recoder.recode('update_feat'):
            self.phi_feat.update(data_list, data_scalar,
                                 sample_list, sample_scalar,
                                 learning_rate=self.cur_lr_feat)

        # update net-phi
        with self.time_recoder.recode('update_net'):
            self.phi_net.update(data_list, data_scalar,
                                sample_list, sample_scalar,
                                learning_rate=self.cur_lr_net)

        # update zeta
        self.norm_const.update(sample_list, learning_rate=self.cur_lr_logz)
        self.norm_const.set_logz1(self.true_logz(self.config.min_len)[0])

        # update simulater
        if self.simulater is not None:
            with self.time_recoder.recode('update_simulater'):
                self.simulater.update(sample_list)

        # update dbg info
        self.sample_cur_pi.fill(0)
        for x in sample_list:
            self.sample_cur_pi[len(x)] += 1
        self.sample_acc_count += self.sample_cur_pi
        self.sample_cur_pi /= self.sample_cur_pi.sum()

        return None

    def initialize(self):
        # print the txt information
        for d, name in zip(self.data.datas, ['train', 'valid', 'test']):
            info = wb.TxtInfo(d)
            print('[TRF]', name, ':', str(info))

        # load prior
        self.priorlm.initialize()

        # create features
        self.phi_feat.initialize()
        self.phi_net.initialize()

        # print parameters
        print('[TRF] feat_num = {:,}'.format(self.phi_feat.get_param_num()))
        print('[TRF] net_num  = {:,}'.format(self.phi_net.get_param_num()))

    def train(self, print_per_epoch=0.1, operation=None):

        # initialize
        self.initialize()

        if self.exist_model():
            self.restore()

        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]
        test_list = self.data.datas[2]

        print('[TRF] [Train]...')
        epoch_contain_step = len(train_list) // self.config.train_batch_size

        time_beginning = time.time()
        model_train_nll = []
        kl_distance = []

        step = self.training_info['trained_step']
        epoch = step / epoch_contain_step
        print_next_epoch = int(epoch)

        while epoch < self.config.max_epoch:

            # update training information
            self.training_info['trained_step'] = step
            self.training_info['trained_epoch'] = epoch
            self.training_info['trained_time'] = (time.time() - time_beginning) / 60

            if step % epoch_contain_step == 0:
                np.random.shuffle(train_list)
                self.save()

            # get empirical list
            data_beg = step % epoch_contain_step * self.config.train_batch_size
            data_list = train_list[data_beg: data_beg + self.config.train_batch_size]

            # draw samples
            with self.time_recoder.recode('sample'):
                sample_list = self.draw(self.config.sample_batch_size)

            # update paramters
            with self.time_recoder.recode('update'):
                # learining rate
                self.cur_lr_feat = self.config.lr_feat.get_lr(step+1, epoch)
                self.cur_lr_net = self.config.lr_net.get_lr(step+1, epoch)
                self.cur_lr_logz = self.config.lr_logz.get_lr(step+1, epoch)
                # update
                self.update(data_list, sample_list)

            ##########################
            # update step
            ##########################
            step += 1
            epoch = step / epoch_contain_step

            # evaulate the nll and KL-distance
            with self.time_recoder.recode('eval_train_nll'):
                model_train_nll.append(self.eval(data_list)[0])
            with self.time_recoder.recode('eval_kl_dis'):
                kl_distance.append(self.simulater.eval(sample_list)[0] - self.eval(sample_list, for_eval=False)[0])

            # print
            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                time_since_beg = (time.time() - time_beginning) / 60

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]
                    # model_test_nll = self.eval(test_list)[0]
                    # simul_valid_nll = self.simulater.eval(session, valid_list)[0]

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                info['lr_feat'] = '{:.2e}'.format(self.cur_lr_feat)
                info['lr_net'] = '{:.2e}'.format(self.cur_lr_net)
                info['lr_logz'] = '{:.2e}'.format(self.cur_lr_logz)
                info['lj_rate'] = self.lj_rate
                info['mv_rate'] = self.mv_rate
                info['train'] = np.mean(model_train_nll[-epoch_contain_step:])
                info['valid'] = model_valid_nll
                # info['test'] = model_test_nll
                info['kl_dis'] = np.mean(kl_distance[-epoch_contain_step:])
                log.print_line(info)

                print('[end]')

                # write time
                f = self.write_files.get('time')
                f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, time_since_beg))
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.time_recoder.items()]) + '\n')
                f.flush()

                #  write zeta, logz, pi
                f = self.write_files.get('zeta')
                f.write('step={}\n'.format(step))
                log.write_array(f, self.sample_cur_pi[self.config.min_len:], name='cur_pi')
                log.write_array(f, self.sample_acc_count[self.config.min_len:]/self.sample_acc_count.sum(), name='all_pi')
                log.write_array(f, self.config.pi_0[self.config.min_len:], name='pi_0  ')
                log.write_array(f, self.norm_const.get_logz(), name='logz  ')

            ###########################
            # extra operations
            ###########################
            if operation is not None:
                operation.run(step, epoch)


class DefaultOps(wb.Operation):
    def __init__(self, m, nbest_list_path, trans_path, ac_score=None):
        self.m = m
        self.nbest_cmp = reader.NBest(nbest_list_path, trans_path, ac_score)
        self.wer_next_epoch = 0
        self.wer_per_epoch = 1.0
        self.write_models = wb.mkdir(os.path.join(self.m.logdir, 'wer_results'))

    def run(self, step, epoch):
        super().run(step, epoch)

        if epoch >= self.wer_next_epoch:
            self.wer_next_epoch = int(epoch + self.wer_per_epoch)

            # resocring
            time_beg = time.time()
            self.nbest_cmp.lmscore = -self.m.get_log_probs(self.nbest_cmp.get_nbest_list(self.m.data))
            rescore_time = time.time() - time_beg

            # compute wer
            time_beg = time.time()
            wer = self.nbest_cmp.wer()
            wer_time = time.time() - time_beg

            wb.WriteScore(self.write_models + '/epoch%.2f' % epoch + '.lmscore', self.nbest_cmp.lmscore)
            print('epoch={:.2f} test_wer={:.2f} lmscale={} '
                  'rescore_time={:.2f}, wer_time={:.2f}'.format(
                   epoch, wer, self.nbest_cmp.lmscale,
                   rescore_time / 60, wer_time / 60))

            res = wb.FRes(os.path.join(self.m.logdir, 'wer_per_epoch.log'))
            res_name = 'epoch%.2f' % epoch
            res.Add(res_name, ['lm-scale'], [self.nbest_cmp.lmscale])
            res.Add(res_name, ['wer'], [wer])





