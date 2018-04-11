import tensorflow as tf
from copy import deepcopy

from base import *
from lm import *


class RJMCMC(object):
    def __init__(self, q_config, gamma, multiple_trial, sample_sub, end_tokens,
                 fun_logps, fun_propose_tag=None,
                 write_files=None,
                 pi_true=None,
                 device='/gpu:0'):
        # simulater
        self.propose_lstm = lstmlm.LM(q_config, device=device, name='rjmcmc_q')
        # length jump probabilities
        self.gamma = gamma
        self.pi_true = pi_true

        self.multiple_trial = multiple_trial
        self.sample_sub = sample_sub
        self.end_tokens = end_tokens

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

    def propose_tag(self, seq_list, pos_vec, is_sample=True):
        """[empyt] donot propose tags"""
        if self.fun_propose_tag is None:
            batch_size = len(seq_list)
            new_tags = np.zeros(batch_size)
            cond_logps = np.zeros(batch_size)
            return new_tags, cond_logps
        else:
            return self.fun_propose_tag(seq_list, pos_vec, is_sample)

    def set_to_state(self, states_list, new_states, pos_vec):
        """

        Args:
            states_list: a list of list of tuple(c,h), states of several sequences
            new_states:  a list of tuple(c,h), state of several sequence at one position

        Returns:

        """
        for i, pos in enumerate(pos_vec):
            if pos == len(states_list[i]):
                states_list[i].append(new_states[i])
            else:
                states_list[i][pos] = new_states[i]  # set the state

    def set_to_seq(self, seq_list, new_w, pos_vec):
        for i, pos in enumerate(pos_vec):
            if pos == len(seq_list[i]):
                seq_list[i].append([new_w[i], 0])  # append a new position
            else:
                seq_list[i].x[0, pos] = new_w[i]   # set the word

    def propose_one(self, seq_list, states_list, pos_vec):
        """
        Args:
            seq_list: a list of seq.Seq()
            states_list: a list of list of lstm-state, corresponding to each sequence
            pos_vec: position

        Returns:
            return the logp and revise the seq_list and states_list
        """

        if isinstance(pos_vec, int):
            pos_vec = np.ones(len(seq_list)) * pos_vec

        input_x = np.array([s.x[0][i-1:i] for s, i in zip(seq_list, pos_vec)])
        try:
            states = [s[i-1] for s, i, in zip(states_list, pos_vec)]
        except IndexError:
            for s, st, i in zip(seq_list, states_list, pos_vec):
                print('s_len={} st_len={}  pos={}'.format(len(s), len(st), i))
            raise IndexError

        # print('states', states)
        y, logp_x, new_states = self.propose_lstm.simulate_h(self.session, input_x, 1, initial_state=states)
        new_x = y[:, -1]

        self.set_to_seq(seq_list, new_x, pos_vec)
        self.set_to_state(states_list, new_states, pos_vec)

        ########################
        # debug
        ########################
        # true_state = self.update_hidden_state(seq_list)
        # f = self.write_files.get('mcmc.dbg')
        # for i, pos in enumerate(pos_vec):
        #     d = np.mean(np.array(states_list[i][pos]) - np.array(true_state[i][pos]))
        #     f.write('[propose] pos={} d={}\n'.format(pos, d))
        #     f.write(str(seq_list[0]) + '\n')
        #     f.write('save={}\ntrue={}\n'.format(states_list[i], true_state[i]))
        ########################

        new_tag, logp_tag = self.propose_tag(seq_list, pos_vec, is_sample=True)
        # set new_tag
        for s, pos, t in zip(seq_list, pos_vec, new_tag):
            s.x[1, pos] = t

        # print('logpx', logp_x)
        # print('logpt', logp_tag)

        return logp_x + logp_tag

    def condition_one(self,  seq_list, states_list, pos_vec):
        if isinstance(pos_vec, int):
            pos_vec = np.ones(len(seq_list)) * pos_vec

        input_x = np.array([s.x[0][i-1:i+1] for s, i in zip(seq_list, pos_vec)])
        states = [s[i-1] for s, i, in zip(states_list, pos_vec)]

        # compute prob and new states
        logp_x, new_states = self.propose_lstm.conditional_h(self.session, input_x, 1, initial_state=states)
        self.set_to_state(states_list, new_states, pos_vec)

        _, logp_tag = self.propose_tag(seq_list, pos_vec, is_sample=False)

        return logp_x + logp_tag

    def propose_seq(self, seq_list, states_list, pos_vec, num_vec=1):
        if isinstance(pos_vec, int):
            pos_vec = [pos_vec] * len(seq_list)
        if isinstance(num_vec, int):
            num_vec = [num_vec] * len(seq_list)
        pos_vec = np.array(pos_vec)
        num_vec = np.array(num_vec)

        final_logp = np.zeros(len(seq_list))
        final_seqs = seq.list_copy(seq_list)
        final_states = deepcopy(states_list)
        cur_pos = np.array(pos_vec)
        for i in range(np.max(num_vec)):
            idx = np.where(num_vec > i)[0]
            logp = self.propose_one(seq_list=[final_seqs[k] for k in idx],
                                    states_list=[final_states[k] for k in idx],
                                    pos_vec=cur_pos[idx])
            final_logp[idx] += logp
            cur_pos += 1

        return final_seqs, final_states, final_logp

    def condition_seq(self, seq_list, states_list, pos_vec, num_vec=1):
        if isinstance(pos_vec, int):
            pos_vec = [pos_vec] * len(seq_list)
        if isinstance(num_vec, int):
            num_vec = [num_vec] * len(seq_list)
        pos_vec = np.array(pos_vec)
        num_vec = np.array(num_vec)

        assert len(seq_list) == len(pos_vec)
        assert len(seq_list) == len(num_vec)
        assert len(seq_list) == len(states_list)

        final_logp = np.zeros(len(seq_list))
        cur_pos = np.array(pos_vec)
        for i in range(np.max(num_vec)):
            idx = np.where(num_vec > i)[0]
            logp = self.condition_one(seq_list=[seq_list[k] for k in idx],
                                      states_list=[states_list[k] for k in idx],
                                      pos_vec=cur_pos[idx])
            final_logp[idx] += logp
            cur_pos += 1

        return final_logp

    def repeat_seq_and_state(self, seqs_list, states_list, lengths):
        seqs_repeat = []
        states_repeat = []
        for s, st, n in zip(seqs_list, states_list, lengths):
            for _ in range(self.multiple_trial):
                seqs_repeat.append(s[0:n])
                states_repeat.append(deepcopy(st)[0:n-1])
        return seqs_repeat, states_repeat

    def local_jump_inc(self, init_seqs, init_states, next_lens):

        init_lens = np.array([len(s) for s in init_seqs])
        jump_rate = np.array(
            [np.log(self.gamma[j, k]) - np.log(self.gamma[k, j]) for j, k in zip(next_lens, init_lens)])

        # repeat the input for multiple trial
        init_seqs_repeat, init_states_repeat = self.repeat_seq_and_state(init_seqs, init_states, init_lens)

        # propose the next sequences multiple times
        next_seqs_repeat, next_states_repeat, next_cond_logp = \
            self.propose_seq(init_seqs_repeat, init_states_repeat,
                             init_lens.repeat(self.multiple_trial) - 1,
                             next_lens.repeat(self.multiple_trial) - init_lens.repeat(self.multiple_trial))

        # add end tokens
        for s in next_seqs_repeat:
            s.append(self.end_tokens)

        # the log prob
        logws = self.fun_logps(next_seqs_repeat)
        logws = logws.reshape((-1, self.multiple_trial))

        # draw next sequence
        draw_idxs = [sp.log_sample(sp.log_normalize(x)) for x in logws]
        draw_idxs_flatten = [i * self.multiple_trial + draw_idxs[i] for i in range(len(draw_idxs))]
        next_seqs = [next_seqs_repeat[i] for i in draw_idxs_flatten]
        next_states = [next_states_repeat[i] for i in draw_idxs_flatten]

        cond_logp = next_cond_logp[draw_idxs_flatten]
        next_logps = logsumexp(logws, axis=-1)
        init_logps = self.fun_logps(init_seqs)

        cur_acc_logps = jump_rate + next_logps - \
                        init_logps - cond_logp - np.log(self.multiple_trial)

        # summary results
        res_dbg = {
            'acc_logps': cur_acc_logps,
            'init_logp': init_logps + np.log(self.multiple_trial),
            'next_logp': next_logps,
            'cond_logp': cond_logp,
            'next_seqs': next_seqs,
            'next_states': next_states,
        }

        return res_dbg

    def local_jump_des(self, init_seqs, init_states, next_lens):
        init_lens = np.array([len(s) for s in init_seqs])
        jump_rate = np.array(
            [np.log(self.gamma[j, k]) - np.log(self.gamma[k, j]) for j, k in zip(next_lens, init_lens)])

        # repeat the input for multiple trial
        next_seqs_repeat, next_states_repeat = self.repeat_seq_and_state(init_seqs, init_states, next_lens)

        # propose
        init_seqs_repeat, init_states_repeat, _ = \
            self.propose_seq(next_seqs_repeat, next_states_repeat,
                             next_lens.repeat(self.multiple_trial) - 1,
                             init_lens.repeat(self.multiple_trial) - next_lens.repeat(self.multiple_trial))

        # add end token
        for s in init_seqs_repeat:
            s.append(self.end_tokens)  # add end-tokens

        # set the initial_sequences
        for i in range(len(init_seqs)):
            init_seqs_repeat[i * self.multiple_trial] = init_seqs[i]
            init_states_repeat[i * self.multiple_trial] = init_states[i]

        logws = self.fun_logps(init_seqs_repeat)
        logws = np.reshape(logws, (-1, self.multiple_trial))
        init_logps = logsumexp(logws, axis=-1)

        # condition logprob
        cond_logp = self.condition_seq(init_seqs, init_states, next_lens - 1, init_lens - next_lens)

        # next sequences
        next_seqs = []
        next_states = []
        for s, st, m in zip(init_seqs, init_states, next_lens):
            ss = s.get_sub(0, m - 1)
            ss.append(self.end_tokens)
            next_seqs.append(ss)
            next_states.append(deepcopy(st[0: m-1]))

        # next logps
        next_logps = self.fun_logps(next_seqs)

        cur_acc_logps = jump_rate + np.log(self.multiple_trial) + next_logps + cond_logp - \
                        init_logps

        # summary results
        res_dbg = {
            'acc_logps': cur_acc_logps,
            'init_logp': init_logps,
            'next_logp': next_logps + np.log(self.multiple_trial),
            'cond_logp': cond_logp,
            'next_seqs': next_seqs,
            'next_states': next_states,
        }
        return res_dbg

    def local_jump(self, seq_list, states_list):

        """
        Args:
            seq_list: a list of seq.Seq()
            states_list: a list of LSTM states of each sequence

        Returns:

        """
        # f = self.write_files.get('mcmc.dbg')
        # f.write('local jump \n')

        batch_size = len(seq_list)

        acc_logps = np.zeros(batch_size)
        init_logp = np.zeros(batch_size)
        next_logp = np.zeros(batch_size)
        cond_logp = np.zeros(batch_size)
        accept = np.array([False] * batch_size)
        next_seqs = [None] * batch_size
        next_states = [None] * batch_size

        # draw lengths
        input_n = np.array([len(x) for x in seq_list])
        next_n = np.array([np.random.choice(len(self.gamma[n]), p=self.gamma[n]) for n in input_n])

        inc_index = np.where(next_n > input_n)[0]
        des_index = np.where(next_n < input_n)[0]
        equ_index = np.where(next_n == input_n)[0]

        def set_res(idx, res):
            acc_logps[idx] = res['acc_logps']
            init_logp[idx] = res['init_logp']
            next_logp[idx] = res['next_logp']
            cond_logp[idx] = res['cond_logp']
            for i, j in enumerate(idx):
                next_seqs[j] = res['next_seqs'][i]
                next_states[j] = res['next_states'][i]

        if len(equ_index) > 0:
            acc_logps[equ_index] = 0
            init_logp[equ_index] = 0
            next_logp[equ_index] = 0
            cond_logp[equ_index] = 0
            for i in equ_index:
                next_seqs[i] = seq_list[i]
                next_states[i] = states_list[i]

        if len(inc_index) > 0:
            init_seqs = [seq_list[i] for i in inc_index]
            init_states = [states_list[i] for i in inc_index]
            final_lens = next_n[inc_index]

            res = self.local_jump_inc(init_seqs, init_states, final_lens)

            set_res(inc_index, res)

        if len(des_index) > 0:
            init_seqs = [seq_list[i] for i in des_index]
            init_states = [states_list[i] for i in des_index]
            final_lens = next_n[des_index]

            res = self.local_jump_des(init_seqs, init_states, final_lens)

            set_res(des_index, res)

        # accept
        res_seqs = seq_list
        res_states = states_list
        for i in range(batch_size):
            self.lj_times += 1
            if sp.accept_logp(acc_logps[i]):
                self.lj_success += 1
                accept[i] = True
                res_seqs[i] = next_seqs[i]
                res_states[i] = next_states[i]

        # set the states length
        for i, s in enumerate(seq_list):
            states_list[i] = states_list[i][0: len(s)-1]  # no end token

        # output debug
        if self.write_files is not None:
            final_n = [len(s) for s in seq_list]
            for i in range(len(seq_list)):
                out_line = '[local jump chain={}] {}->{} {} acc_logp={:.2f} acc={} '.format(
                    i, input_n[i], next_n[i], final_n[i], float(acc_logps[i]), accept[i]
                )
                out_line += 'logp_x={:.2f} logp_y={:.2f} logp_g={:.2f} '.format(
                    init_logp[i], next_logp[i], cond_logp[i])
                out_line += 'acc_rate={:.2f}% '.format(100.0 * self.lj_success / self.lj_times)
                out_line += '[{}/{}] '.format(self.lj_success, self.lj_times)

                f = self.write_files.get('markov')
                f.write(out_line + '\n')
                f.flush()

        return res_seqs, res_states

    def markov_move_sub(self, seq_list, states_list, sub_pos, sub_len):
        chain_num = len(seq_list)

        # repeat the input for multiple trial
        next_seqs_repeat = []
        next_states_repeat = []
        for s, st in zip(seq_list, states_list):
            for _ in range(self.multiple_trial):
                next_seqs_repeat.append(s.copy())
                next_states_repeat.append(deepcopy(st))

        # multiple trial propose the new sequences
        next_seqs_repeat, next_states_repeat, next_cond_logp = \
            self.propose_seq(next_seqs_repeat, next_states_repeat,
                             sub_pos,
                             sub_len.repeat(self.multiple_trial))

        # compute the logw
        next_logw_repeat = self.fun_logps(next_seqs_repeat) - next_cond_logp

        # sample the next seqs
        draw_idxs = [sp.log_sample(sp.log_normalize(x)) for x in next_logw_repeat.reshape((chain_num, -1))]
        draw_idxs_flatten = [i * self.multiple_trial + draw_idxs[i] for i in range(len(draw_idxs))]
        next_seqs = [next_seqs_repeat[i] for i in draw_idxs_flatten]
        next_states = [next_states_repeat[i] for i in draw_idxs_flatten]

        # draw reference x
        # as is independence sampling
        # there is no need to generate new samples
        init_logw = self.fun_logps(seq_list) - self.condition_seq(seq_list, states_list, sub_pos, sub_len)
        init_logw_repeat = np.array(next_logw_repeat)
        init_logw_repeat[draw_idxs_flatten] = init_logw

        # compute the acceptance rate
        next_weight = logsumexp(next_logw_repeat.reshape((chain_num, -1)), axis=-1)
        init_weight = logsumexp(init_logw_repeat.reshape((chain_num, -1)), axis=-1)
        acc_logps = next_weight - init_weight

        # acceptance
        acc_probs = np.exp(np.minimum(0., acc_logps))
        accept = acc_probs >= np.random.uniform(size=acc_probs.shape)

        res_seqs = []
        res_states = []
        for i, acc in enumerate(accept):
            if acc:
                res_seqs.append(next_seqs[i])
                res_states.append(next_states[i])
            else:
                res_seqs.append(seq_list[i])
                res_states.append(states_list[i])

        # debug informations
        if self.write_files is not None:
            self.mv_times += accept.size
            self.mv_success += accept.sum()
            out_line = '[Markov move] acc=' + log.to_str(acc_probs) + \
                       ' weight_y=' + log.to_str(next_weight) + \
                       ' weight_x=' + log.to_str(init_weight)
            f = self.write_files.get('markov')
            f.write(out_line + '\n')
            f.flush()

        return res_seqs, res_states

    def markov_move(self, seq_list, states_list):
        # f = self.write_files.get('mcmc.dbg')
        # f.write('markov move\n')

        seq_len = np.array([len(s) for s in seq_list])
        max_len = np.max(seq_len)
        sub_sent = self.sample_sub if self.sample_sub > 0 else 1

        # skip the beg / end tokens
        for beg_pos in range(1, max_len-1, sub_sent):
            # find all the sequence whose length is larger than beg_pos
            idx = np.where(seq_len-1 > beg_pos)[0]
            # desicide the end position
            sub_len = np.minimum(sub_sent, seq_len-1-beg_pos)
            res_seqs, res_states = self.markov_move_sub(seq_list=[seq_list[i] for i in idx],
                                                        states_list=[states_list[i] for i in idx],
                                                        sub_pos=beg_pos, sub_len=sub_len[idx])

            for i, s, st in zip(idx, res_seqs, res_states):
                seq_list[i] = s
                states_list[i] = st

        return seq_list, states_list

    def update(self, sample_list):
        """update the propose lstm model"""
        x_list = [s.x[0].tolist() for s in sample_list]
        self.propose_lstm.sequence_update(self.session, x_list)

    def update_hidden_state(self, seq_list):
        x_batch, n_batch = reader.produce_data_to_array([s.x[0, 0:-1] for s in seq_list])  # remove the end-token
        states = self.propose_lstm.get_hidden_states(self.session, x_batch, n_batch)
        return states

    def eval(self, seq_list):
        x_list = [s.x[0].tolist() for s in seq_list]
        return self.propose_lstm.eval(self.session, x_list)

    def hidden_states_diff(self, slist1, slist2):
        res = []
        for s1, s2 in zip(slist1, slist2):
            d = [np.mean(np.abs(np.array(a) - np.array(b))) for a, b in zip(s1, s2)]
            res.append(d)
        return res

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
