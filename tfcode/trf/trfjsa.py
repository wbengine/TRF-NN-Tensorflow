from .trfbase import *
from scipy.misc import logsumexp

from base import *
from lm import *


class Config(BaseConfig):
    def __init__(self, data):
        super().__init__(data)
        # for sampling
        self.auxiliary_model = 'lstm'  # one of (lstm, lstm-block, ngram)
        self.auxiliary_config = lstmlm.Config(data)   # the config of auxiliary LSTM
        # self.auxiliary_shortlist = None
        # self.auxiliary_hidden = 250
        # self.auxiliary_lr = 1.0
        # self.auxiliary_opt = 'sgd'
        self.jump_width = 5
        self.chain_num = 10
        self.multiple_trial = 10
        self.sample_sub = 5
        # for learning
        self.opt_method = 'adam'
        self.batch_size = self.chain_num * self.multiple_trial
        self.train_batch_size = 10 * self.batch_size
        self.sample_batch_size = self.batch_size
        self.max_epoch = 10
        self.max_grad_norm = 100.0
        self.lr_cnn = LearningRateTime(1.0, 1.0, None, 0, 1e-4)
        self.lr_param = LearningRateTime(1.0, 1.0, None, 0, 1e-3)
        self.lr_zeta = LearningRateTime(1.0, 0.2, 2000, 0, 1.0)
        self.zeta_gap = 10
        self.var_gap = 1e-12  # variance gap for learning
        self.L2_reg = 0
        self.dropout = 0


class Simulater(object):
    """
    Simulater is a auxiliary distribution used in Joint SA algorithms
    It is used to propose a sequence based on the current seqeunce,
    which helps to accelerate the MCMC sampling in SA algorithms
    """
    def __init__(self):
        # recode the time cost
        self.time_recoder = wb.clock()

    def update(self, session, sample_mcmc):
        """
        update the simulater parameters
        Args:
            session: tf.Session()
            sample_mcmc: a list of mcmc state, such as [(inputs, lengths), (inputs, lengths), ...]
                sample_mcmc[0] is the initial state, which is same to sample_mcmc[-1] at the previous iteration

        Returns:
            None
        """
        pass

    def prepare_mcmc(self, sample_mcmc):
        """
        prepare the mcmc state.
        [(inputs_0, lengths_0), (inputs_1, lengths_1), ..., (inputs_K, lengths_K)]
        to
        [seqs_0, seqs_1, ..., seqs_K-1], [seqs_1, seqs_2, ..., seqs_K]
        where seqs_i is a sequence list indicating (inputs_i, lengths_i)
        """
        def extract_seqs_nd(tuple_list):
            seqs = []
            for inputs, lengths in tuple_list:
                seqs += reader.extract_data_from_trf(inputs, lengths)
            return seqs

        return extract_seqs_nd(sample_mcmc[0: -1]), extract_seqs_nd(sample_mcmc[1:])

    def local_jump_propose(self, session, inputs, lengths, next_lengths):
        """
        Args:
            session: tf.Session()
            inputs: current sequences
            lengths: np.array length of current sequences
            next_lengths: np.array length of next length > lengths

        Returns:
            nexts
        """
        pass

    def local_jump_condition(self, session, inputs, lengths, next_lengths):
        """next_lengths < lengths"""
        pass

    def markov_move_propose(self, session, inputs, lengths, beg_pos, end_pos):
        pass

    def markov_move_condition(self, session, inputs, lengths, beg_pos, end_pos):
        pass

    def eval(self, session, seq_list):
        pass


class SimulaterLSTM(Simulater):
    def __init__(self, config, device, name='simulater'):
        super().__init__()
        self.model = lstmlm.LM(config, device=device, name=name)

        self.context_size = 5
        self.seqs_cache = []
        self.write_files = wb.FileBank('simulater')

    def update(self, session, sample_mcmc):
        with self.time_recoder.recode('update'):
            for x in self.prepare_mcmc(sample_mcmc)[1]:
                self.seqs_cache.append(x[1:])  # rm the beg-tokens

            # self.model.sequence_update(session, seq_list)

            x_list, y_list = reader.produce_data_for_rnn(self.seqs_cache,
                                                         self.model.config.batch_size,
                                                         self.model.config.step_size,
                                                         include_residual_data=False)
            if len(x_list) > 0:
                for x, y in zip(x_list, y_list):
                    self.model.update(session, x, y)

                self.seqs_cache = []

    def local_jump_propose(self, session, inputs, lengths, next_lengths):
        if isinstance(lengths, int):
            lengths = np.array([lengths] * len(inputs))
        if isinstance(next_lengths, int):
            next_lengths = np.array([next_lengths] * len(inputs))

        assert np.all(next_lengths > lengths)

        max_next_lenght = np.max(next_lengths)
        y_list = []
        logp_list = []
        beg = 0
        for i in range(1, len(inputs)+1):
            if i == len(inputs) or lengths[i] != lengths[beg]:
                input_len = lengths[beg]
                y, y_logp = self.model.simulate(session, inputs[beg:i, 0:input_len],
                                                next_lengths[beg: i] - input_len,
                                                initial_state=True, context_size=self.context_size)
                y_list.append(np.pad(y, [[0, 0], [0, max_next_lenght-y.shape[1]]], 'edge'))
                logp_list.append(y_logp)

                beg = i

        res_y = np.concatenate(y_list, axis=0)
        res_logp = np.concatenate(logp_list, axis=0)

        return res_y, res_logp

    def local_jump_condition(self, session, inputs, lengths, next_lengths):
        if isinstance(lengths, int):
            lengths = np.array([lengths] * len(inputs))
        if isinstance(next_lengths, int):
            next_lengths = np.array([next_lengths] * len(inputs))

        assert np.all(next_lengths < lengths)

        return self.model.conditional(session, inputs, next_lengths, lengths,
                                      initial_state=True, context_size=self.context_size)

    def markov_move_propose(self, session, inputs, lengths, beg_pos, end_pos):
        sample_nums = np.minimum(lengths, end_pos)-beg_pos
        nexts, logps = self.model.simulate(session, inputs[:, 0: beg_pos], sample_nums,
                                           initial_state=True, context_size=self.context_size)
        nexts = np.concatenate([nexts, inputs[:, nexts.shape[1]:]], axis=-1)
        return nexts, logps

    def markov_move_condition(self, session, inputs, lengths, beg_pos, end_pos):
        return self.model.conditional(session, inputs, beg_pos, np.minimum(lengths, end_pos),
                                      initial_state=True, context_size=self.context_size)

    def eval(self, session, seq_list):
        """return (NLL, PPL)"""
        with self.time_recoder.recode('eval'):
            res = self.model.eval(session, seq_list, net=self.model.valid_net)
        return res


class SimulaterNgram(Simulater):
    def __init__(self, trf_config, device=None, name='simulater'):
        super().__init__()
        self.vocab_size = trf_config.vocab_size
        self.class_size = int(np.sqrt(self.vocab_size))
        self.class_num = int(np.ceil(self.vocab_size / self.class_size))
        self.prob_size = int(self.class_num * self.class_size)
        self.bigram = np.zeros((self.vocab_size, self.prob_size))
        self.bigram[:, 0: self.vocab_size] = 1.0 / self.vocab_size
        self.bigram_class = np.sum(self.bigram.reshape((self.vocab_size, self.class_num, self.class_size)),
                                   axis=-1)
        self.learn_rate = 0.1

        print('[SimulaterNgram]\n')
        print('\tvocab_size=%d\n\tclass_num=%d\n\tclass_size=%d\n\tprob_size=%d\n' % (
            self.vocab_size, self.class_num, self.class_size, self.prob_size))

        self.time_recoder = wb.clock()

    def update(self, session, seq_list):
        with self.time_recoder.recode('update'):
            bigram_update = dict()
            # count
            for seq in seq_list:
                for i in range(len(seq) - 1):
                    count = bigram_update.setdefault(seq[i], np.zeros(self.prob_size))
                    count[seq[i+1]] += 1
            # normalize and update
            for context, count in bigram_update.items():
                count /= count.sum()
                self.bigram[context] += self.learn_rate * (count - self.bigram[context])
                self.bigram_class[context] = np.sum(self.bigram[context].reshape((self.class_num, self.class_size)),
                                                    axis=-1)

    def propose_batch(self, session, initial_seqs, n):
        with self.time_recoder.recode('propose'):
            batch_size = len(initial_seqs)
            initial_len = len(initial_seqs[0])
            final_seqs = np.concatenate((initial_seqs, np.zeros((batch_size, n))), axis=-1).astype('int32')
            final_logps = np.zeros(batch_size)
            for i in range(initial_len, initial_len + n):
                for j in range(batch_size):
                    pc = self.bigram_class[final_seqs[j][i-1]]
                    pw = self.bigram[final_seqs[j][i-1]].reshape((self.class_num, self.class_size))
                    # propose c
                    c = np.random.choice(self.class_num, p=pc)
                    # propose w
                    w = np.random.choice(self.class_size, p=pw[c] / pc[c])
                    final_seqs[j, i] = w + self.class_size * c
                    final_logps[j] += np.log(pw[c, w])

            return final_seqs, final_logps

    def conditional_batch(self, session, input_x, input_n, pos):
        with self.time_recoder.recode('condition'):
            final_logps = []
            for x, n in zip(input_x, input_n):
                logp = 0
                for i in range(pos, n):
                    logp += np.log(self.bigram[x[i-1], x[i]])
                final_logps.append(logp)
        return np.array(final_logps)

    def propose_one(self, session, prefix, n):
        x_batch, logp_batch = self.propose_batch(session, [prefix], n)
        return list(x_batch[0]), logp_batch[0]

    def conditional_one(self, session, seq, pos):
        return self.conditional_batch(session,
                                      np.array(seq).reshape(1, -1),
                                      np.array([len(seq)]),
                                      pos)

    def eval(self, session, seq_list):
        with self.time_recoder.recode('eval'):
            cost = 0
            word_num = 0
            sent_num = 0
            for seq in seq_list:
                cost += -np.sum(np.log([self.bigram[seq[i-1], seq[i]] for i in range(1, len(seq))]))
                word_num += len(seq) - 1
                sent_num += 1
            nll = cost / sent_num
            ppl = np.exp(cost / word_num)
        return nll, ppl


class FastTRF(TRFFrame):
    def __init__(self, config, data, name='TRF', logdir='trf', simulater_device='/gpu:0'):
        super().__init__(config, data, name=name, logdir=logdir)

        # auxiliary simulater for sampling
        self.create_simulater(simulater_device)

        # summary
        self.scalar_bank = layers.SummaryScalarBank(['epoch_train', 'epoch_sample',
                                                     'lr_cnn', 'lr_param', 'lr_zeta',
                                                     'nll_train', 'nll_valid', 'nll_test',
                                                     'ppl_train', 'ppl_valid', 'ppl_test',
                                                     'nll_true_train', 'nll_true_valid',
                                                     'kl_distance',
                                                     'pi_distance',
                                                     'LocalJumpRate', 'MarkovMoveRate',
                                                     'param_norm', 'grad_norm',
                                                     'wer'])
        self.vars_history = layers.SummaryVariables(tf.trainable_variables())

        self._global_step = None
        self._pure_summary = None
        # the session used to calculate tensorflow operations
        self.session = None

    def create_simulater(self, simulater_device):
        if simulater_device is None:
            self.simulater = None
            return

        if self.config.auxiliary_model == 'lstm':
            self.simulater = SimulaterLSTM(self.config.auxiliary_config, device=simulater_device)
        elif self.config.auxiliary_model == 'ngram':
            self.simulater = SimulaterNgram(self.config)
        else:
            raise TypeError('unknown auxiliary model = {}'.format(self.config.auxiliary_model))

    def initialize(self, session):
        self.set_session(session)

    def set_session(self, session):
        self.session = session

    def get_session(self):
        if self.session is None:
            raise ValueError('self.session is None. please call self.set_session() to set the session!!')
        return self.session

    def local_jump(self, x):
        k = len(x)  # old length
        j = sp.linear_sample(self.gamma[k])  # new length

        assert (j >= self.config.min_len)
        assert (j <= self.config.max_len)

        # local jump
        if j == k:
            return x

        end_tokens = np.ones((self.config.multiple_trial, 1), dtype='int32') * self.config.end_token

        if j > k:
            x_repeat = np.tile(x, [self.config.multiple_trial, 1])
            y_repeat, g_repeat = self.simulater.local_jump_propose(self.get_session(), x_repeat[:, 0:-1], k-1, j-1)
            y_repeat = np.concatenate([y_repeat, end_tokens], axis=-1)  # append end tokens
            logw = self.get_log_probs(y_repeat.tolist())
            new_idx = sp.log_sample(sp.log_normalize(logw))

            new_x = y_repeat[new_idx].tolist()
            g_add = g_repeat[new_idx]
            assert len(new_x) == j

            logp_new = logsumexp(logw)
            logp_old = np.log(self.config.multiple_trial) + self.get_log_prob(x)

            acc_logp = np.log(self.gamma[j, k]) - np.log(self.gamma[k, j]) + \
                       logp_new - (logp_old + g_add)

        elif j < k:
            x_repeat = np.tile(x, [self.config.multiple_trial, 1])
            y_repeat, _ = self.simulater.local_jump_propose(self.get_session(), x_repeat[:, 0:j-1], j-1, k-1)
            y_repeat = np.concatenate([y_repeat, end_tokens], axis=-1)  # append end tokens
            y_repeat[-1] = x
            logw = self.get_log_probs(y_repeat.tolist())

            g_add = self.simulater.local_jump_condition(self.get_session(),
                                                        np.array([x[0:k-1]]), k-1, j-1)[0]
            new_x = x[0:j-1]
            new_x.append(x[-1])
            assert len(new_x) == j

            logp_new = np.log(self.config.multiple_trial) + self.get_log_prob(new_x)
            logp_old = sp.log_sum(logw)

            acc_logp = np.log(self.gamma[j, k]) - np.log(self.gamma[k, j]) + \
                       logp_new + g_add - logp_old

        else:
            raise Exception('Invalid jump form {} to {}'.format(k, j))

        self.lj_times += 1
        if sp.accept_logp(acc_logp):
            self.lj_success += 1
        else:
            new_x = list(x)

        out_line = '[local jump] {}->{} acc_logp={:.2f} '.format(k, j, float(acc_logp))
        out_line += 'logp_new={:.2f} logp_old={:.2f} '.format(logp_new, logp_old)
        out_line += 'g_add={:.2f} '.format(float(g_add))
        out_line += ' jump_rate={:.2f}% '.format(100.0 * self.lj_rate)
        out_line += ' acc_rate={:.2f}% '.format(100.0 * self.lj_success / self.lj_times)
        out_line += '[{}/{}] '.format(self.lj_success, self.lj_times)
        f = self.write_files.get('markov')
        f.write(out_line + '\n')
        f.flush()

        return new_x

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
            y_multiple, logp_y_multiple = self.simulater.local_jump_propose(self.get_session(),
                                                                            x_repeat, n_repeat-1, m_repeat-1)
            # add end-tokens
            y_multiple = np.pad(y_multiple, [[0, 0], [0, 1]], 'constant')
            y_multiple[np.arange(y_multiple.shape[0]), m_repeat-1] = self.config.end_token
            logw_multiple_y = self.logps(y_multiple, m_repeat)

            draw_idxs = [sp.log_sample(sp.log_normalize(x)) for x in logw_multiple_y.reshape((chain_num, -1))]
            draw_idxs_flatten = [i * self.config.multiple_trial + draw_idxs[i] for i in range(len(draw_idxs))]
            new_y = y_multiple[draw_idxs_flatten]
            new_m = m_repeat[draw_idxs_flatten]
            g_add = logp_y_multiple[draw_idxs_flatten]
            assert np.all(new_m == final_lens)

            cur_acc_logps = cur_jump_rate + \
                            logsumexp(logw_multiple_y.reshape((chain_num, -1)), axis=-1) - \
                            np.log(self.config.multiple_trial) - g_add - self.logps(init_seqs, init_lens)

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
            x_multiple, logp_x_multiple = self.simulater.local_jump_propose(self.get_session(),
                                                                            y_repeat, m_repeat-1, n_repeat-1)
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
            logw_multiple_x = self.logps(x_multiple, n_repeat)

            g_add = self.simulater.local_jump_condition(self.get_session(),
                                                        init_seqs, init_lens-1, final_lens-1)
            new_y = np.array(init_seqs)
            new_m = final_lens
            new_y[np.arange(new_y.shape[0]), new_m-1] = self.config.end_token

            cur_acc_logps = cur_jump_rate + \
                            np.log(self.config.multiple_trial) + g_add + self.logps(new_y, new_m) - \
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
        multiple_y, logp_multiple_y = self.simulater.markov_move_propose(self.get_session(),
                                                                         x_repeat, n_repeat-1,
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
                                         self.simulater.markov_move_condition(self.get_session(),
                                                                              input_x, input_n-1,
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

        out_line = '[Markov move] acc=' + to_str(acc_probs) + \
                   ' weight_y=' + to_str(weight_y) + \
                   ' weight_x=' + to_str(weight_x)
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

    def sample(self, input_x, input_n):
        # with self.time_recoder.recode('sample'):
        #     input_x, input_n = self.trans_dim_MH(input_x, input_n)

        with self.time_recoder.recode('local_jump'):
            input_x, input_n = self.local_jump_batch(input_x, input_n)

        with self.time_recoder.recode('markov_move'):
            input_x, input_n = self.markov_move_batch(input_x, input_n)

        return input_x, input_n

    def update(self, session, train_seqs, sample_seqs, global_step, global_epoch):
        lr_param = self.config.lr_param.get_lr(global_step, global_epoch)
        lr_zeta = self.config.lr_zeta.get_lr(global_step, global_epoch)

        train_scalars = 1.0 / len(train_seqs) * np.ones(len(train_seqs))
        sample_n = np.array([len(x) for x in sample_seqs])
        sample_scalars = 1.0 / len(sample_seqs) * self.config.pi_true[sample_n] / self.config.pi_0[sample_n]

        # update zeta
        self.update_zeta(sample_seqs, lr_zeta)

        # update simulater
        if self.simulater is not None:
            with self.time_recoder.recode('update_simulater'):
                self.simulater.update(self.get_session(), self.sample_mcmc)

        # update discrete parameters
        if self.feat_word is not None:
            with self.time_recoder.recode('update_feat'):
                self.feat_word.seq_update(train_seqs, train_scalars, sample_seqs, sample_scalars,
                                          lr=lr_param,
                                          L2_reg=self.config.L2_reg,
                                          dropout=self.config.dropout)
        if self.feat_class is not None:
            with self.time_recoder.recode('update_feat'):
                self.feat_class.seq_update(self.data.seqs_to_class(train_seqs), train_scalars,
                                           self.data.seqs_to_class(sample_seqs), sample_scalars,
                                           lr=lr_param,
                                           L2_reg=self.config.L2_reg,
                                           dropout=self.config.dropout)

        return {'lr_param': lr_param, 'lr_zeta': lr_zeta}

    def load(self, logname):
        """load all the parameters"""
        self.load_model(logname)
        self.load_feat(logname)

    def save(self, logname):
        """save all the parameters"""
        self.save_model(logname)
        self.save_feat(logname)

    def prepare(self):
        self.precompute_feat(self.data.datas[0])

    def train_after_update(self, **argv):
        pass

    def train(self, sv, session, nbest=None,
              print_per_epoch=0.,
              wer_per_epoch=1.,
              lmscore_per_epoch=1,
              model_per_epoch=50,
              load_model_epoch=None,
              eval_list=None,
              operation=None):
        """training framework"""
        self.set_session(session)
        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]
        test_list = self.data.datas[2]
        print('[TRF] train_list={:,}'.format(len(train_list)),
              'valid_list={:,}'.format(len(valid_list)))

        print('[TRF] load parameters...')
        logname = os.path.join(self.logdir, self.name)
        wb.mkdir(os.path.join(self.logdir, 'lmscore'))
        wb.mkdir(os.path.join(self.logdir, 'models'))
        logname_lmscore = os.path.join(self.logdir, 'lmscore/' + self.name)
        logname_models = os.path.join(self.logdir, 'models/' + self.name)

        if load_model_epoch is None:
            # load the last model
            self.load(logname)
        else:
            self.load(logname_models + '.epoch{}'.format(load_model_epoch))

        print('[TRF] training prepare...')
        self.prepare()

        print('[TRF] [Train]...')
        epoch_contain_step = len(train_list) // self.config.train_batch_size
        print_next_epoch = 0
        wer_next_epoch = 0

        time_beginning = time.time()
        model_train_nll = []
        true_train_nll = []
        kl_distance = []

        if self._global_step is not None:
            step = session.run(self._global_step)
        else:
            step = 0
        epoch = step * self.config.train_batch_size / len(train_list)

        while epoch < self.config.max_epoch:

            if step % epoch_contain_step == 0:
                np.random.shuffle(train_list)

            # get empirical list
            empirical_beg = step % epoch_contain_step * self.config.train_batch_size
            empirical_list = train_list[empirical_beg: empirical_beg + self.config.train_batch_size]

            ########################
            # draw samples
            ########################
            with self.time_recoder.recode('sample'):
                sample_list = self.draw(self.config.sample_batch_size)

            ###########################
            # update paramters
            ###########################
            with self.time_recoder.recode('update'):
                lr_for_all = self.update(session, empirical_list, sample_list, step+1, epoch)

            ##########################
            # update step
            ##########################
            step += 1
            epoch = step * self.config.train_batch_size / len(train_list)

            ##########################
            # evaulate the nll and KL-distance
            ##########################
            with self.time_recoder.recode('eval_train_nll'):
                model_train_nll.append(self.eval(empirical_list)[0])
            with self.time_recoder.recode('eval_kl_dis'):
                if self.simulater is not None:
                    kl_distance.append(self.simulater.eval(session, sample_list)[0] - self.eval_pi0(sample_list)[0])
                else:
                    kl_distance.append(0)

            # write summary
            self.scalar_bank.write_summary(sv, session, 'epoch_train', epoch, step)
            for name, v in lr_for_all.items():
                self.scalar_bank.write_summary(sv, session, name, v, step)
            self.scalar_bank.write_summary(sv, session, 'nll_train', model_train_nll[-1], step)
            self.scalar_bank.write_summary(sv, session, 'pi_distance', self.pi_distance(), step)
            self.scalar_bank.write_summary(sv, session, 'kl_distance', kl_distance[-1], step)
            self.scalar_bank.write_summary(sv, session, 'LocalJumpRate', self.lj_rate, step)
            self.scalar_bank.write_summary(sv, session, 'MarkovMoveRate', self.mv_rate, step)
            if self._pure_summary is not None:
                sv.summary_computed(session, session.run(self._pure_summary), step)

            ####################
            # external process
            ####################
            self.train_after_update(sv=sv, session=session, global_step=step,
                                    eval_list=eval_list)

            ###########################
            # print
            ###########################
            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                time_since_beg = (time.time() - time_beginning) / 60

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]
                    model_test_nll = self.eval(test_list)[0]
                    # simul_valid_nll = self.simulater.eval(session, valid_list)[0]

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                for name, v in lr_for_all.items():
                    info[name] = to_str(v, '{:.1e}')
                info['lj_rate'] = self.lj_rate
                info['mv_rate'] = self.mv_rate
                info['train'] = np.mean(model_train_nll[-epoch_contain_step:])
                info['valid'] = model_valid_nll
                info['test'] = model_test_nll
                # info['simu_valid'] = simul_valid_nll
                info['kl_dis'] = np.mean(kl_distance[-epoch_contain_step:])
                print_line(info)
                self.scalar_bank.write_summary(sv, session, 'nll_valid', model_valid_nll, step)
                self.scalar_bank.write_summary(sv, session, 'nll_test', model_test_nll, step)

                ######################################
                # calculate the WER
                #####################################
                if epoch >= wer_next_epoch and nbest is not None:
                    wer_next_epoch = int(epoch + wer_per_epoch)

                    self.time_recoder.beg()
                    nbest.lmscore = -self.get_log_probs(nbest.get_nbest_list(self.data))
                    if lmscore_per_epoch is not None and int(epoch) % lmscore_per_epoch == 0:
                        wb.WriteScore(logname_lmscore + '.epoch{}.lmscore'.format(int(epoch)), nbest.lmscore)
                    else:
                        wb.WriteScore(logname_lmscore + '.lmscore', nbest.lmscore)
                    time_wer = self.time_recoder.end()
                    wer = nbest.wer()
                    print('wer={:.2f} lmscale={:.2f} score_time={:.2f}'.format(wer, nbest.lmscale, time_wer),
                          end=' ', flush=True)
                    self.scalar_bank.write_summary(sv, session, 'wer', wer, step)

                #########################
                # Write zeta and pi
                #########################
                logz_sams = np.array(self.logz)
                logz_true = np.zeros_like(logz_sams)

                if self.config.max_len <= 5 and self.config.vocab_size < 100:
                    with self.time_recoder.recode('true_eval'):
                        self.true_normalize_all()
                        true_train_nll.append(self.eval(empirical_list))
                        print('train(true)=' + to_str(np.mean(true_train_nll[-epoch_contain_step:], axis=0)),
                              'valid(true)=' + to_str(self.eval(valid_list)),
                              end=' '
                              )
                        logz_true = np.array(self.logz)
                        self.logz = np.array(logz_sams)
                        self.zeta = np.array(logz_sams - logz_sams[self.config.min_len])

                print('[end]')

                #############################
                # write log
                #############################
                #  write zeta, logz, pi
                self.logwrite_pi(logz_sams, logz_true)
                # write time info
                f = self.write_files.get('time')
                f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, (time.time() - time_beginning)/60))
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.items()]) + ' ')
                if self.simulater is not None:
                    f.write(' '.join(['simulater_{}={:.2f}'.format(x[0], x[1]) for x in self.simulater.time_recoder.items()]))
                f.write('\n')
                f.flush()

            ###########################
            # extra operations
            ###########################
            if operation is not None:
                operation.run(step, epoch)

            # write feat after each epoch
            if step % epoch_contain_step == 0:
                self.save(logname)
                if model_per_epoch is not None and int(epoch) % model_per_epoch == 0:
                    self.save(logname_models + '.epoch{}'.format(int(epoch)))


from .trfnce import NoiseSamplerNgram


class FastTRF2(FastTRF):
    def __init__(self, config, data, name='TRF', logdir='trf', simulater_device='/gpu:0'):
        super().__init__(config, data, name, logdir, simulater_device=None)

        self.noise_sampler = NoiseSamplerNgram(config, data, 3)

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
        input_x, input_n = self.trans_dim_MH(input_x, input_n)
        return input_x, input_n
