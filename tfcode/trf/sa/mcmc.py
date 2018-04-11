import tensorflow as tf
from copy import deepcopy

from base import *
from lm import *


class RJMCMC(object):
    def __init__(self, simulater, gamma, multiple_trial, sample_sub, end_token,
                 fun_logps, fun_propose_tag=None,
                 write_files=None,
                 pi_true=None):
        # simulater
        self.simulater = simulater
        # length jump probabilities
        self.gamma = gamma
        self.pi_true = pi_true

        self.multiple_trial = multiple_trial
        self.sample_sub = sample_sub
        self.end_token = end_token

        # cmp the logps
        self.fun_logps = fun_logps  # input a list of seq.Seq(), output np.array of the logps
        self.fun_propose_tag = fun_propose_tag

        # debug variables
        self.lj_times = 1
        self.lj_success = 0
        self.lj_rate = None
        self.mv_times = 1
        self.mv_success = 0
        self.mv_rate = None

        self.write_files = write_files

    @property
    def session(self):
        return tf.get_default_session()

    def local_jump(self, input_x, input_n):

        """
        Args:
            seq_list: a list of seq.Seq()
            states_list: a list of LSTM states of each sequence

        Returns:

        """
        batch_size = len(input_x)
        old_seqs = reader.extract_data_from_trf(input_x, input_n)
        new_seqs = [None] * batch_size
        acc_logps = np.zeros(batch_size)

        next_n = np.array([np.random.choice(len(self.gamma[n]), p=self.gamma[n]) for n in input_n])
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

            x_repeat = init_seqs.repeat(self.multiple_trial, axis=0)
            n_repeat = init_lens.repeat(self.multiple_trial, axis=0)
            m_repeat = final_lens.repeat(self.multiple_trial, axis=0)
            y_multiple, logp_y_multiple = self.simulater.local_jump_propose(x_repeat, n_repeat - 1, m_repeat - 1)
            # add end-tokens
            y_multiple = np.pad(y_multiple, [[0, 0], [0, 1]], 'constant')
            y_multiple[np.arange(y_multiple.shape[0]), m_repeat - 1] = self.end_token
            logw_multiple_y = self.fun_logps(y_multiple, m_repeat)

            draw_idxs = [sp.log_sample(sp.log_normalize(x)) for x in logw_multiple_y.reshape((chain_num, -1))]
            draw_idxs_flatten = [i * self.multiple_trial + draw_idxs[i] for i in range(len(draw_idxs))]
            new_y = y_multiple[draw_idxs_flatten]
            new_m = m_repeat[draw_idxs_flatten]
            g_add = logp_y_multiple[draw_idxs_flatten]
            assert np.all(new_m == final_lens)

            cur_acc_logps = cur_jump_rate + \
                            logsumexp(logw_multiple_y.reshape((chain_num, -1)), axis=-1) - \
                            np.log(self.multiple_trial) - g_add - self.fun_logps(init_seqs, init_lens)

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

            y_repeat = init_seqs.repeat(self.multiple_trial, axis=0)
            n_repeat = init_lens.repeat(self.multiple_trial, axis=0)
            m_repeat = final_lens.repeat(self.multiple_trial, axis=0)
            x_multiple, logp_x_multiple = self.simulater.local_jump_propose(y_repeat, m_repeat - 1, n_repeat - 1)
            # add end-token
            x_multiple = np.pad(x_multiple, [[0, 0], [0, 1]], 'constant')
            x_multiple[np.arange(x_multiple.shape[0]), n_repeat - 1] = self.end_token
            # set the initial_sequences
            for i in range(chain_num):
                # if len(x_multiple[i * self.config.multiple_trial]) != len(init_seqs[i]):
                #     print(x_multiple[i * self.config.multiple_trial])
                #     print(init_seqs[i])
                n = n_repeat[i * self.multiple_trial]
                x_multiple[i * self.multiple_trial, 0: n] = init_seqs[i, 0: init_lens[i]]
            logw_multiple_x = self.fun_logps(x_multiple, n_repeat)

            g_add = self.simulater.local_jump_condition(init_seqs, init_lens - 1, final_lens - 1)
            new_y = np.array(init_seqs)
            new_m = final_lens
            new_y[np.arange(new_y.shape[0]), new_m - 1] = self.end_token

            cur_acc_logps = cur_jump_rate + \
                            np.log(self.multiple_trial) + g_add + self.fun_logps(new_y, new_m) - \
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
        x_repeat = input_x.repeat(self.multiple_trial, axis=0).astype(input_x.dtype)
        n_repeat = input_n.repeat(self.multiple_trial, axis=0).astype(input_n.dtype)

        # propose y
        multiple_y, logp_multiple_y = self.simulater.markov_move_propose(x_repeat, n_repeat-1,
                                                                         beg_pos, end_pos)
        # return logp_multiple_y
        multiple_y[np.arange(len(multiple_y)), n_repeat-1] = self.end_token  # set end tokens
        logw_multiple_y = self.fun_logps(multiple_y, n_repeat) - logp_multiple_y
        # logw_multiple_y = self.phi(multiple_y, n_repeat) -\
        #                   self.simulater.markov_move_condition(self.get_session(),
        #                                                        x_repeat, n_repeat-1, multiple_y,
        #                                                        beg_pos, end_pos)

        # sample y
        draw_idxs = [sp.log_sample(sp.log_normalize(x)) for x in logw_multiple_y.reshape((chain_num, -1))]
        draw_idxs_flatten = [i * self.multiple_trial + draw_idxs[i] for i in range(len(draw_idxs))]
        new_y = multiple_y[draw_idxs_flatten]

        # draw reference x
        # as is independence sampling
        # there is no need to generate new samples
        logw_multiple_x = np.array(logw_multiple_y)
        logw_multiple_x[draw_idxs_flatten] = self.fun_logps(input_x, input_n) - \
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

    def markov_move(self, input_x, input_n):

        max_len = np.max(input_n)
        sub_sent = self.sample_sub if self.sample_sub > 0 else max_len

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

    def update(self, sample_list):
        """update the propose lstm model"""
        self.simulater.update(sample_list)

    def eval(self, seq_list):
        return self.simulater.eval(seq_list)

    def reset_dbg(self):
        self.lj_times = 0
        self.lj_success = 0
        self.mv_times = 0
        self.mv_success = 0

    def update_dbg(self):

        def interpolate(old_rate, new_rate):
            if old_rate is None:
                return new_rate
            return 0.9 * old_rate + 0.1 * new_rate

        if self.lj_times > 0:
            self.lj_rate = interpolate(self.lj_rate, self.lj_success / self.lj_times)

        if self.mv_times > 0:
            self.mv_rate = interpolate(self.mv_rate, self.mv_success / self.mv_times)


class MCMC(RJMCMC):
    def local_jump(self, seq_list, states_list):
        init_len = np.array([len(s) for s in seq_list])
        next_len = np.random.choice(len(self.pi_true), size=len(seq_list), p=self.pi_true)

        inc_index = np.where(next_len > init_len)[0]
        des_index = np.where(next_len < init_len)[0]
        equ_index = np.where(next_len == init_len)[0]

        next_seqs = [None] * len(seq_list)
        next_states = [None] * len(states_list)
        if len(equ_index) > 0:
            for i in equ_index:
                next_seqs[i] = seq_list[i]
                next_states[i] = states_list[i]

        if len(inc_index) > 0:
            local_seqs, local_states, _ = self.propose_seq(seq_list=[seq_list[i] for i in inc_index],
                                                           states_list=[states_list[i] for i in inc_index],
                                                           pos_vec=init_len[inc_index] - 1,
                                                           num_vec=next_len[inc_index] - init_len[inc_index])

            for s in local_seqs:
                s.append(self.end_tokens)

            for i, s, st in zip(inc_index, local_seqs, local_states):
                next_seqs[i] = s
                next_states[i] = st

        if len(des_index) > 0:
            for i in des_index:
                next_seqs[i] = seq_list[i][0: next_len[i]-1]
                next_seqs[i].append(self.end_tokens)
                next_states[i] = states_list[i][0: next_len[i]-1]

        return next_seqs, next_states
