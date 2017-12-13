import tensorflow as tf
from copy import deepcopy
import numpy as np
import time
import json
import os
from collections import OrderedDict

from .net_lstm import *
from . import layers
from . import reader
from . import trfbase
from . import wblib as wb
from . import word2vec
from . import sampling as sp
from . import net_trf_cnn
from . import net_trf_cnn2
from . import net_trf_cnn_char
from . import net_trf_rnn
from . import net_trf_rnn_char
from . import net_trf_tree


trf_network_maps = {
    'cnn': net_trf_cnn,
    'rnn': net_trf_rnn,
    'rnn_char': net_trf_rnn_char,
    'cnn_char': net_trf_cnn_char,
    'tree': net_trf_tree,
}


class Config(trfbase.BaseConfig):
    def __init__(self, data, trf_structure='cnn'):
        super().__init__(data)
        self.trf_structure = trf_structure
        if trf_structure in trf_network_maps:
            self.config_trf = trf_network_maps[trf_structure].NetTRF.Config(data)
        else:
            raise TypeError('undefined trf structure name = ' + trf_structure)

        # lstm for proposal in JSA
        self.config_lstm = NetLSTM.Config(data)

        # for sampling
        self.jump_width = 5
        self.chain_num = 10
        self.multiple_trial = 10
        self.sample_sub = 5

        # for learining
        self.batch_size = self.chain_num * self.multiple_trial
        self.max_epoch = 100
        self.lr_cnn = trfbase.LearningRateTime(1.0, 1.0, None, 0, 1e-4)
        self.lr_param = trfbase.LearningRateTime(1.0, 1.0, None, 0, 1e-3)
        self.lr_zeta = trfbase.LearningRateTime(1.0, 0.2, 2000, 0, 1.0)
        self.L2_reg = 0


class Net(object):
    def __init__(self, config, is_training=True, name='global', reuse=None):
        """

        Args:
            config: the configuration
            is_training: if True, then compute the gradient of potential function with respect to parameters
            name: the name
            reuse: if resuse the variables
        """
        self.is_training = is_training
        self.name = name

        self.config = config
        self.min_len = config.config_trf.min_len
        self.max_len = config.config_trf.max_len
        self.len_num = self.max_len - self.min_len + 1

        # create variables
        with tf.name_scope(self.name), tf.variable_scope(self.name, reuse=reuse):
            self.gamma = tf.get_variable('gamma', [self.len_num, self.len_num], tf.float32,
                                         trainable=False,
                                         initializer=tf.constant_initializer(
                                             sp.len_jump_distribution(0, self.len_num-1, config.jump_width)),
                                         )
            print('trf-net=', self.config.trf_structure)
            self.net_lstm = NetLSTM(config.config_lstm, reuse=reuse)
            self.net_trf = trf_network_maps[self.config.trf_structure].NetTRF(self.config.config_trf,
                                                                              reuse=reuse,
                                                                              propose_lstm=self.net_lstm)

            # inputs: of shape (batch_size, seq_len)
            # lengths: of shape (batch_size,)
            self._inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
            self._lengths = tf.placeholder(tf.int32, [None], name='lengths')
            self.draw, self.draw_dbgs = self.get_draw(self._inputs, self._lengths)

            # debugs
            self.local_jump_test = self.get_local_jump(self._inputs, self._lengths)
            self.markov_move_test = self.get_markov_move(self._inputs, self._lengths)
            self._beg_pos = tf.placeholder(tf.int32, [])
            self._end_pos = tf.placeholder(tf.int32, [])
            self._initial_states = self.net_lstm.cell.zero_state(tf.shape(self._inputs)[0], tf.float32)
            self.markov_move_sub_test = self.get_markov_move_sub(self._inputs, self._lengths,
                                                                 self._beg_pos, self._end_pos,
                                                                 self._initial_states)

    def get_local_jump(self, _inputs, _lengths):
        with tf.name_scope('local_jump'):
            # draw next lengths
            curr_len_idx = _lengths - self.min_len
            next_len_idx = layers.tf_random_choice(self.len_num, tf.gather(self.gamma, curr_len_idx))
            # next_len_idx = curr_len_idx - 1
            jump_prob_to_next = tf.gather_nd(self.gamma, tf.stack([curr_len_idx, next_len_idx], axis=1))
            jump_prob_to_cur = tf.gather_nd(self.gamma, tf.stack([next_len_idx, curr_len_idx], axis=1))
            jump_rate = tf.log(jump_prob_to_cur) - tf.log(jump_prob_to_next)

            next_lengths = next_len_idx + self.min_len

            # pad two fake sequences
            init_lengths = tf.concat([_lengths, [self.min_len, self.min_len+1]], axis=0)
            next_lengths = tf.concat([next_lengths, [self.min_len+1, self.min_len]], axis=0)
            jump_rate = tf.pad(jump_rate, [[0, 2]])

            max_buf = tf.reduce_max(tf.maximum(init_lengths, next_lengths))
            init_seqs = tf.pad(_inputs, [[0, 2], [0, max_buf-tf.shape(_inputs)[1]]])
            chain_num = tf.shape(init_seqs)[0]

            inc_mask = tf.greater(next_lengths, init_lengths)
            des_mask = tf.less(next_lengths, init_lengths)
            equ_mask = tf.logical_not(tf.logical_or(inc_mask, des_mask))

            # inc
            inc_acc, inc_seqs, inc_lens = self.get_local_jump_inc(tf.boolean_mask(init_seqs, inc_mask),
                                                                  tf.boolean_mask(init_lengths, inc_mask),
                                                                  tf.boolean_mask(next_lengths, inc_mask),
                                                                  tf.boolean_mask(jump_rate, inc_mask))
            indices = tf.cast(tf.where(inc_mask), tf.int32)
            inc_seqs = tf.scatter_nd(indices, inc_seqs, [chain_num, tf.shape(inc_seqs)[1]])
            inc_seqs = tf.pad(inc_seqs, [[0, 0], [0, max_buf - tf.shape(inc_seqs)[1]]])
            inc_acc = tf.scatter_nd(indices, inc_acc, [chain_num])
            inc_lens = tf.scatter_nd(indices, inc_lens, [chain_num])

            #dec
            des_acc, des_seqs, des_lens = self.get_local_jump_des(tf.boolean_mask(init_seqs, des_mask),
                                                                  tf.boolean_mask(init_lengths, des_mask),
                                                                  tf.boolean_mask(next_lengths, des_mask),
                                                                  tf.boolean_mask(jump_rate, des_mask))
            indices = tf.cast(tf.where(des_mask), tf.int32)
            des_seqs = tf.scatter_nd(indices, des_seqs, [chain_num, tf.shape(des_seqs)[1]])
            des_seqs = tf.pad(des_seqs, [[0, 0], [0, max_buf - tf.shape(des_seqs)[1]]])
            des_acc = tf.scatter_nd(indices, des_acc, [chain_num])
            des_lens = tf.scatter_nd(indices, des_lens, [chain_num])

            # equal
            equ_seqs = init_seqs * tf.expand_dims(tf.cast(equ_mask, dtype=init_seqs.dtype), axis=-1)
            equ_acc = tf.zeros([chain_num], dtype=tf.float32)
            equ_lens = init_lengths * tf.cast(equ_mask, dtype=init_lengths.dtype)

            # summary
            final_acc = inc_acc + des_acc + equ_acc
            final_seqs = inc_seqs + des_seqs + equ_seqs
            final_lengths = inc_lens + des_lens + equ_lens

            # acceptance
            acc_mask = tf.greater_equal(tf.exp(final_acc), tf.random_uniform([chain_num]))
            acc_succeed = tf.reduce_sum(tf.cast(acc_mask[0: -2], tf.int32))
            res_seqs = tf.where(acc_mask, final_seqs, init_seqs)
            res_lengths = tf.where(acc_mask, final_lengths, init_lengths)

            # remove the fake sequence
            max_res_len = tf.reduce_max(res_lengths)
            res_seqs = res_seqs[0: -2, 0: max_res_len]
            res_lengths = res_lengths[0: -2]

            # return dbg
            dbgs = OrderedDict([
                ('init_len', init_lengths),
                ('next_len', next_lengths),
                ('acc_logp', final_acc),
                ('acc_bool', acc_mask),
                ('succeed', acc_succeed),
                ('times', chain_num),
                ('inc_mask', inc_mask),
                ('des_mask', des_mask),
                ('equ_mask', equ_mask),
            ])

            return res_seqs, res_lengths, dbgs

    def get_local_jump_inc(self, init_seqs, init_lengths, final_lengths, jump_rate):
        with tf.name_scope('local_jump_inc'):
            chain_num = tf.shape(init_seqs)[0]

            x_repeat = layers.repeat(init_seqs, self.config.multiple_trial, axis=0)
            n_repeat = layers.repeat(init_lengths, self.config.multiple_trial, axis=0)
            m_repeat = layers.repeat(final_lengths, self.config.multiple_trial, axis=0)
            y_multiple, _, logp_y_multiple, _ = self.net_lstm.get_draw(None, x_repeat, n_repeat-1, m_repeat-n_repeat)
            # pad and add end-tokens
            y_multiple = layers.append_sequence(y_multiple, m_repeat-1, self.config.end_token)
            # compute trf logprobs
            logw_y_multiple = self.net_trf.get_logp(y_multiple, m_repeat)
            # draw y based logw
            logps = tf.reshape(logw_y_multiple, [chain_num, self.config.multiple_trial])  # reshape
            logps = logps - tf.reduce_logsumexp(logps, axis=-1, keep_dims=True)  # normalize
            draw_idxs = layers.tf_random_choice(self.config.multiple_trial, tf.exp(logps))  # draw
            # select y and g_add
            draw_idxs += tf.range(chain_num) * self.config.multiple_trial  # flatten
            new_y = tf.gather(y_multiple, draw_idxs)
            new_m = tf.gather(m_repeat, draw_idxs)
            g_add = tf.gather(logp_y_multiple, draw_idxs)

            # compute tha acceptance probs
            logw_sum = tf.reshape(logw_y_multiple, [chain_num, self.config.multiple_trial])
            logw_sum = tf.reduce_logsumexp(logw_sum, axis=-1)

            acc_logps = jump_rate + logw_sum - \
                        np.log(self.config.multiple_trial) - g_add - self.net_trf.get_logp(init_seqs, init_lengths)

            return acc_logps, new_y, new_m

    def get_local_jump_des(self, init_seqs, init_lengths, final_lengths, jump_rate):
        with tf.name_scope('local_jump_des'):
            chain_num = tf.shape(init_seqs)[0]

            y_repeat = layers.repeat(init_seqs, self.config.multiple_trial, axis=0)
            n_repeat = layers.repeat(init_lengths, self.config.multiple_trial, axis=0)
            m_repeat = layers.repeat(final_lengths, self.config.multiple_trial, axis=0)
            x_multiple, _, _, _ = self.net_lstm.get_draw(None, y_repeat, m_repeat-1, n_repeat-m_repeat)
            # pad and add end-tokens, [chain_num*multiple_trial, max(n_repeat)]
            x_multiple = layers.append_sequence(x_multiple, n_repeat-1, self.config.end_token)
            # set the initial sequences
            maxlen = tf.reduce_max(init_lengths)
            x_temp = tf.reshape(x_multiple, [chain_num, self.config.multiple_trial, -1])
            x_temp = tf.concat([x_temp[:, 0:-1, :], tf.reshape(init_seqs[:, 0:maxlen], [chain_num, 1, -1])], axis=1)
            x_multiple = tf.reshape(x_temp, [chain_num * self.config.multiple_trial, -1])
            # compute trf logprobs
            logw = self.net_trf.get_logp(x_multiple, n_repeat)
            logw_sum = tf.reshape(logw, [chain_num, self.config.multiple_trial])
            logw_sum = tf.reduce_logsumexp(logw_sum, axis=-1)

            # new sequences
            new_y = layers.append_sequence(init_seqs[:, 0:-1], final_lengths-1, self.config.end_token)
            new_m = final_lengths
            g_add, _ = self.net_lstm.get_logp(None, init_seqs, final_lengths-1, init_lengths-1)
            logp_new_y = self.net_trf.get_logp(new_y, new_m)

            # acceptance
            acc_logps = jump_rate + np.log(self.config.multiple_trial) + g_add + logp_new_y - logw_sum

            return acc_logps, new_y, new_m

    def get_markov_move(self, _inputs, _lengths, name='markov_move'):
        with tf.name_scope(name):
            def body(seqs, beg_pos, states, acc_times, acc_succeed):

                mask = tf.greater(_lengths-1, beg_pos)
                indices = tf.cast(tf.where(mask), tf.int32)
                part_seqs = tf.gather(seqs, tf.reshape(indices, [-1]))
                part_lengths = tf.gather(_lengths, tf.reshape(indices, [-1]))
                part_states = self.net_lstm.get_state_gather(states, tf.reshape(indices, [-1]))

                part_seqs, part_states, acc = self.get_markov_move_sub(part_seqs, part_lengths,
                                                                       beg_pos, beg_pos+self.config.sample_sub,
                                                                       part_states)

                # set the sequences back to the whole batches
                res_seqs = tf.scatter_nd(indices, part_seqs, tf.shape(seqs))
                res_seqs += seqs * tf.expand_dims(tf.cast(tf.logical_not(mask), tf.int32), axis=-1)
                # set the states backs to the whole batches
                res_states = []
                for (part_c, part_h), (total_c, total_h) in zip(part_states, states):
                    c = tf.scatter_nd(indices, part_c, tf.shape(total_c))
                    c += total_c * tf.expand_dims(tf.cast(tf.logical_not(mask), tf.float32), axis=-1)
                    h = tf.scatter_nd(indices, part_h, tf.shape(total_h))
                    h += total_h * tf.expand_dims(tf.cast(tf.logical_not(mask), tf.float32), axis=-1)
                    res_states.append(tf.contrib.rnn.LSTMStateTuple(c, h))
                res_states = tuple(res_states)

                acc_times += tf.shape(acc)[0]
                acc_succeed += tf.reduce_sum(tf.cast(acc, tf.int32))

                return res_seqs, beg_pos + self.config.sample_sub, res_states, acc_times, acc_succeed

            def cond(seqs, beg_pos, states, acc_times, acc_succeed):
                return tf.reduce_any(tf.greater(_lengths - 1, beg_pos))

            initial_states = self.net_lstm.cell.zero_state(tf.shape(_inputs)[0], tf.float32)
            # state_shape = tuple([tf.contrib.rnn.LSTMStateTuple(c.get_shape(), h.get_shape()) for c, h in initial_states])
            final_seqs, _, _, acc_times, acc_succeed = tf.while_loop(cond, body,
                                                                     loop_vars=[_inputs, 1, initial_states, 0, 0],
                                                                     )

            # final_seqs, _, _, acc_times, acc_succeed = body(_inputs, 1, initial_states, 0, 0)


            dbgs = {'times': acc_times, 'succeed': acc_succeed}

            return final_seqs, _lengths, dbgs

    def get_markov_move_sub(self, seqs, lengths, beg_pos, end_pos, initial_state=None, name='markov_move_sub'):
        with tf.name_scope(name):
            chain_num = tf.shape(seqs)[0]
            max_len = tf.shape(seqs)[1]

            x_repeat = layers.repeat(seqs, self.config.multiple_trial, axis=0)
            n_repeat = layers.repeat(lengths, self.config.multiple_trial, axis=0)

            # propose y
            beg_pos_vec = tf.ones([chain_num * self.config.multiple_trial], dtype=tf.int32) * beg_pos
            end_pos_vec = tf.minimum(n_repeat - 1, end_pos)
            if initial_state is None:
                # using all the history words
                multiple_y, _, logp_multiple_y, final_state_y = self.net_lstm.get_draw(initial_state, x_repeat,
                                                                                       beg_pos_vec,
                                                                                       end_pos_vec - beg_pos_vec)
            else:
                # repeat initial_state
                state_repeat = self.net_lstm.get_state_repeat(initial_state, self.config.multiple_trial)

                # using the initial_state
                multiple_y, _, logp_multiple_y, final_state_y = self.net_lstm.get_draw(state_repeat,
                                                                                       x_repeat[:, beg_pos-1: beg_pos],
                                                                                       beg_pos_vec - beg_pos + 1,
                                                                                       end_pos_vec - beg_pos_vec)
                # concate the head of sequences
                multiple_y = tf.concat([x_repeat[:, 0: beg_pos-1], multiple_y], axis=-1)

            # append the rest parts
            multiple_y = tf.pad(multiple_y, [[0, 0], [0, max_len - tf.shape(multiple_y)[1]]])
            head_mask = tf.sequence_mask(end_pos_vec, max_len)
            tail_mask = tf.logical_not(head_mask)
            multiple_y = multiple_y * tf.cast(head_mask, tf.int32) + x_repeat * tf.cast(tail_mask, tf.int32)

            # draw y
            logw_multiple_y = self.net_trf.get_logp(multiple_y, n_repeat) - logp_multiple_y
            logps = tf.reshape(logw_multiple_y, [chain_num, -1])
            logps -= tf.reduce_logsumexp(logps, axis=-1, keep_dims=True)
            draw_idxs = layers.tf_random_choice(self.config.multiple_trial, tf.exp(logps))
            # select y
            draw_idxs += tf.range(chain_num) * self.config.multiple_trial  # flatten
            new_y = tf.gather(multiple_y, draw_idxs)
            logw_y = tf.gather(logw_multiple_y, draw_idxs)
            final_state_y = self.net_lstm.get_state_gather(final_state_y, draw_idxs)
            # return logw_y, new_y
            if initial_state is None:
                logp_x, final_state_x = self.net_lstm.get_logp(initial_state, seqs,
                                                               tf.ones([chain_num], dtype=tf.int32)*beg_pos,
                                                               tf.minimum(lengths - 1, end_pos))
            else:
                logp_x, final_state_x = self.net_lstm.get_logp(initial_state, seqs[:, beg_pos-1:],
                                                               tf.ones([chain_num], dtype=tf.int32),
                                                               tf.minimum(lengths - 1, end_pos) - beg_pos + 1)
            logw_x = self.net_trf.get_logp(seqs, lengths) - logp_x

            # logw summation
            logw_sum_y = tf.reduce_logsumexp(tf.reshape(logw_multiple_y, [chain_num, -1]), axis=-1)
            logw_sum_x = logw_sum_y - logw_y + logw_x
            acc_logp = logw_sum_y - logw_sum_x

            # accept
            accept = tf.greater_equal(tf.exp(acc_logp), tf.random_uniform(tf.shape(acc_logp)))
            # accept the sequences
            res_seqs = tf.where(accept, new_y, seqs)
            # accept the final_state
            res_state = []
            for (cx, hx), (cy, hy) in zip(final_state_x, final_state_y):
                c = tf.where(accept, cy, cx)
                h = tf.where(accept, hy, hx)
                res_state.append(tf.contrib.rnn.LSTMStateTuple(c, h))
            res_state = tuple(res_state)

            return res_seqs, res_state, accept

    def get_draw(self, seqs, lengths):
        draw_dbgs = OrderedDict()

        seqs, lengths, local_jump_dbgs = self.get_local_jump(seqs[:, 0: tf.reduce_max(lengths)], lengths)
        for name, v in local_jump_dbgs.items():
            draw_dbgs['local_jump/'+name] = v

        seqs, lengths, markov_move_dbgs = self.get_markov_move(seqs[:, 0: tf.reduce_max(lengths)], lengths)
        for name, v in markov_move_dbgs.items():
            draw_dbgs['markov_move/'+name] = v

        return [seqs, lengths], draw_dbgs

    def run_draw(self, session, inputs, lengths):
        res, dbgs = session.run([self.draw, self.draw_dbgs], {self._inputs: inputs, self._lengths: lengths})
        # return (nexts, lengths), dbgs
        return res, dbgs


class DistributedNet(object):
    def __init__(self,  config, is_training=True, devices=['/gpu:0'], name='distributed', reuse=None):
        self.config = config
        self.is_training = is_training
        self.name = name

        self.nets = []
        with tf.device(devices[0]), tf.name_scope(name):
            for i, dev in enumerate(devices):
                with tf.device(dev):
                    print('dev %d on %s' % (i, dev))
                    net = Net(config, is_training, name=name, reuse=reuse if i == 0 else True)
                    self.nets.append(net)

            self.net_trf = self.nets[0].net_trf
            self.net_lstm = self.nets[0].net_lstm

            # for draw
            self.draw_list = [net.draw for net in self.nets]
            self.draw_dbgs_list = [net.draw_dbgs for net in self.nets]

    def run_draw(self, session, inputs, lengths):
        # split the inputs and lengths to different devices
        feed_dict = {}
        chain_num = len(inputs)
        chain_per_device = int(np.ceil(chain_num / len(self.nets)))
        for i in range(len(self.nets)):
            beg = i * chain_per_device
            end = beg + chain_per_device
            feed_dict[self.nets[i]._inputs] = inputs[beg: end]
            feed_dict[self.nets[i]._lengths] = lengths[beg: end]
        res_list, dbgs_list = session.run([self.draw_list, self.draw_dbgs_list], feed_dict)

        all_seqs = []
        # merge sequences
        for res in res_list:
            all_seqs += reader.extract_data_from_trf(*res)

        # merge dbgs
        all_dbgs = None
        for dbg in dbgs_list:
            if all_dbgs is None:
                all_dbgs = dbg
            else:
                for key in all_dbgs:
                    if isinstance(all_dbgs[key], np.ndarray):
                        all_dbgs[key] = np.concatenate([all_dbgs[key], dbg[key]], axis=0)
                    else:
                        all_dbgs[key] += dbg[key]

        return reader.produce_data_to_trf(all_seqs), all_dbgs


class Operation(object):
    def __init__(self, trf_model):
        self.m = trf_model
        self.step = 0
        self.epoch = 0

    def run(self, step, epoch):
        self.step = step
        self.epoch = epoch


class TRF(trfbase.TRFFrame):
    def __init__(self, config, data, name='trf', logdir='trf', device='/gpu:0'):
        super().__init__(config, data, name=name, logdir=logdir)

        if not isinstance(device, list):
            with tf.device(device):
                self.train_net = Net(config, True)
        else:
            self.train_net = DistributedNet(config, True, device)

        self.net_trf = self.train_net.net_trf
        self.net_lstm = self.train_net.net_lstm

        self.global_steps = self.net_trf.global_step
        self.param_size = tf.add_n([tf.size(v) for v in self.net_trf.vars])
        self.param_norm = tf.global_norm(self.net_trf.vars)
        self.summ_vars = layers.SummaryVariables()

        # super().create_simulater(device)
        self.session = None

        # saver
        self.saver = tf.train.Saver()
        self.is_load_model = False

        # summary
        self.scalar_bank = layers.SummaryScalarBank(['epoch_train', 'epoch_sample',
                                                     'lr_nn', 'lr_param', 'lr_zeta',
                                                     'nll_train', 'nll_valid', 'nll_test',
                                                     'ppl_train', 'ppl_valid', 'ppl_test',
                                                     'nll_true_train', 'nll_true_valid',
                                                     'kl_distance',
                                                     'pi_distance',
                                                     'LocalJumpRate', 'MarkovMoveRate',
                                                     'param_norm', 'grad_norm',
                                                     'wer'])

    def set_session(self, session):
        self.session = session

    def get_session(self):
        if self.session is None:
            raise ValueError('self.session is None. please call self.set_session() to set the session!!')
        return self.session

    def phi(self, inputs, lengths):
        feat_weight = super().phi(inputs, lengths)  # weight of discrete features
        return self.net_trf.run_phi(self.get_session(), inputs, lengths)

    def logps(self, inputs, lengths):
        feat_weight = super().phi(inputs, lengths)  # weight of discrete features
        return self.net_trf.run_logps(self.get_session(), inputs, lengths)

    def get_log_probs(self, seq_list, is_norm=True):
        batch_size = self.config.batch_size

        splited_seq_list, splited_index = self.data.cut_data_to_length(seq_list, self.config.max_len)
        logprobs = np.zeros(len(splited_seq_list))

        if is_norm:
            for i in range(0, len(splited_seq_list), batch_size):
                logprobs[i: i + batch_size] = self.logps(
                    *reader.produce_data_to_trf(splited_seq_list[i: i + batch_size])
                )
        else:
            # assert len(seq_list) == len(splited_seq_list)
            for i in range(0, len(splited_seq_list), batch_size):
                logprobs[i: i + batch_size] = self.phi(
                    *reader.produce_data_to_trf(splited_seq_list[i: i + batch_size])
                )

        # merge the logprobs
        res_logps = np.array([np.sum(logprobs[i: j]) for i, j in splited_index])

        return res_logps

    def sample(self, input_x, input_n):

        res, all_dbgs = self.train_net.run_draw(self.get_session(), input_x, input_n)

        # update acceptance rate
        self.lj_success += all_dbgs.setdefault('local_jump/succeed', 0)
        self.lj_times += all_dbgs.setdefault('local_jump/times', 0)
        self.mv_success += all_dbgs.setdefault('markov_move/succeed', 0)
        self.mv_times += all_dbgs.setdefault('markov_move/times', 0)

        # write to files
        f = self.write_files.get('markov')
        for name, v in all_dbgs.items():
            f.write('{} = {}\n'.format(name, v))
        f.flush()

        return res

    def true_normalize_all(self):
        super().true_normalize_all()
        self.net_trf.set_logz_base(self.get_session(), self.logz[self.config.min_len])
        self.net_trf.set_zeta(self.get_session(), self.zeta)

    def update_zeta(self, sample_list, lr_zeta):
        # update zeta
        sample_pi = np.zeros(self.config.max_len + 1)
        for seq in sample_list:
            sample_pi[len(seq)] += 1.
        sample_pi /= len(sample_list)

        self.sample_acc_count += sample_pi * len(sample_list)
        self.sample_cur_pi = sample_pi

        self.logz = self.net_trf.get_logz(self.get_session())
        self.zeta = self.net_trf.get_zeta(self.get_session())

    def update(self, train_seqs, sample_seqs, global_step):
        # assert len(train_seqs) % self.config.batch_size == 0
        # assert len(sample_seqs) % self.config.batch_size == 0

        lr_param = self.config.lr_param.get_lr(global_step)
        lr_zeta = self.config.lr_zeta.get_lr(global_step)
        lr_cnn = self.config.lr_cnn.get_lr(global_step)

        # set lr
        self.net_trf.run_set_lr(self.get_session(), lr_cnn, lr_zeta)

        # update parameters
        inputs, lengths = reader.produce_data_to_trf(train_seqs + sample_seqs)
        with self.time_recoder.recode('trf_update'):
            self.net_trf.run_train(self.get_session(), inputs, lengths, len(train_seqs))

        # update logz_base
        self.net_trf.set_logz_base(self.get_session(), self.true_normalize(self.config.min_len))

        # update super
        with self.time_recoder.recode('lstm_update'):
            # self.net_trf.run_copy_vars(self.get_session())  # using trf's parameters
            self.net_lstm.run_train(self.get_session(), *reader.produce_data_to_trf(sample_seqs))

        # update zeta
        self.update_zeta(sample_seqs, lr_zeta)
        # super().update(self.get_session(), train_seqs, sample_seqs, global_step)

        return {'lr_nn': lr_cnn, 'lr_zeta': lr_zeta}

    def eval(self, data_list):
        self.net_trf.set_pi(self.get_session(), self.config.pi_true)
        logps = self.get_log_probs(data_list)
        self.net_trf.set_pi(self.get_session(), self.config.pi_0)

        lens = [len(x) - int(self.config.beg_token is not None) for x in data_list]
        s = - sum(logps)
        nll = s / len(data_list)
        ppl = np.exp(s / sum(lens))
        return nll, ppl

    def eval_pi0(self, data_list, is_norm=True):
        logps = self.get_log_probs(data_list, is_norm)

        lens = [len(x) - int(self.config.beg_token is not None) for x in data_list]
        s = - sum(logps)
        nll = s / len(data_list)
        ppl = np.exp(s / sum(lens))
        return nll, ppl, logps

    def save_exist(self, logname, is_pretrain=False):
        if is_pretrain:
            return wb.exists(logname + '.pretrain.ckpt.index')
        else:
            return wb.exists(logname + '.ckpt.index')

    def save(self, logname, is_pretrain=False):
        """save mode to dirs"""
        # super().save(logname)
        print('[TRF] save ckpt to %s' % logname)
        if is_pretrain:
            self.saver.save(self.get_session(), logname + '.pretrain.ckpt')
        else:
            self.saver.save(self.get_session(), logname + '.ckpt')

    def load(self, logname):
        """save mode to dirs"""
        # super().load(logname)
        if wb.exists(logname + '.ckpt.index'):
            print('[TRF] load ckpt from %s' % logname)
            self.saver.restore(self.get_session(), logname + '.ckpt')
            self.is_load_model = True

        elif wb.exists(logname + '.pretrain.ckpt.index'):
            print('[TRF] load pretrain.ckpt from %s' % logname)
            self.saver.restore(self.get_session(), logname + '.pretrain.ckpt')
            self.net_trf.run_copy_vars(self.get_session())
            self.is_load_model = True

    def prepare(self):
        # super().prepare()
        print('[TRF] nn.param_num={:,}'.format(self.get_session().run(self.param_size)))

    def pre_train(self, sv, session, batch_size=100, max_epoch=10, lr=1e-3):
        self.set_session(session)

        logname = os.path.join(self.logdir, self.name)
        if self.save_exist(logname, is_pretrain=True):
            print('pretrain model exist, skip pretrain!')
            return

        train_list = self.data.datas[0]
        eval_list = self.data.datas[1]
        train_nll = []
        eval_next_epoch = 0
        time_beginning = time.time()

        for epoch in range(max_epoch):
            for seqs, precent in reader.produce_batch_seqs(train_list, batch_size, output_precent=True):
                inputs, lengths = reader.produce_data_to_trf(seqs)

                loss = self.net_trf.run_pre_train(session, inputs, lengths, lr=lr)
                train_nll.append(loss)

                cur_epoch = epoch + precent

                if cur_epoch > eval_next_epoch:
                    eval_next_epoch += 0.1

                    info = OrderedDict()
                    info['epoch'] = cur_epoch
                    info['loss'] = np.average(train_nll[-len(train_list)//batch_size:]) / batch_size
                    info['time'] = (time.time() - time_beginning) / 60
                    trfbase.print_line(info)
                    print('[end]')

            eval_nll, eval_ppl = self.net_trf.run_pre_train_eval(session, eval_list, batch_size)
            info = OrderedDict()
            info['epoch'] = epoch + 1
            info['valid_nll'] = eval_nll
            info['valid_ppl'] = eval_ppl
            trfbase.print_line(info)
            print('[end]')

            # write to files
            self.save(logname, is_pretrain=True)
            self.net_trf.run_copy_vars(session)

    def train(self, sv, session, nbest=None, nbest_list=None,
              print_per_epoch=0.,
              wer_per_epoch=1.,
              lmscore_per_epoch=1,
              model_per_epoch=50,
              load_model_epoch=None,
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
        train_batch_size = self.config.config_trf.train_batch_size
        sample_batch_size = self.config.config_trf.sample_batch_size
        epoch_contain_step = len(train_list) // train_batch_size

        time_beginning = time.time()
        model_train_nll = []
        true_train_nll = []
        saved_sample_list = []

        step = session.run(self.global_steps)
        epoch = step * train_batch_size / len(train_list)
        print_next_epoch = np.floor(epoch)
        wer_next_epoch = np.floor(epoch)

        while epoch < self.config.max_epoch:

            if step % epoch_contain_step == 0:
                np.random.shuffle(train_list)

            # get empirical list
            empirical_beg = step % epoch_contain_step * train_batch_size
            empirical_list = train_list[empirical_beg: empirical_beg + train_batch_size]

            ########################
            # draw samples
            ########################
            with self.time_recoder.recode('sample'):
                sample_list = self.draw(sample_batch_size)
                saved_sample_list += sample_list
            # print('end_sample')

            ###########################
            # update paramters
            ###########################
            with self.time_recoder.recode('update'):
                lr_for_all = self.update(empirical_list, sample_list, step+1)
                # print('end_update')

            ##########################
            # update step
            ##########################
            step += 1
            epoch = step * train_batch_size / len(train_list)

            ##########################
            # evaulate the nll and KL-distance
            ##########################
            with self.time_recoder.recode('eval_train_nll'):
                model_train_nll.append(self.eval(empirical_list)[0])

            # write summary
            self.scalar_bank.write_summary(sv, session, 'epoch_train', epoch, step)
            for name, v in lr_for_all.items():
                self.scalar_bank.write_summary(sv, session, name, v, step)
            self.scalar_bank.write_summary(sv, session, 'nll_train', model_train_nll[-1], step)
            self.scalar_bank.write_summary(sv, session, 'pi_distance', self.pi_distance(), step)
            self.scalar_bank.write_summary(sv, session, 'LocalJumpRate', self.lj_rate, step)
            self.scalar_bank.write_summary(sv, session, 'MarkovMoveRate', self.mv_rate, step)

            ###########################
            # print
            ###########################
            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                time_since_beg = (time.time() - time_beginning) / 60

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]
                    model_test_nll = self.eval(test_list)[0]
                    model_valid_nll_pi0 = self.eval_pi0(valid_list)[0]
                    simul_valid_nll = self.net_lstm.run_eval(session, valid_list)[0]

                with self.time_recoder.recode('eval_kl_dis'):
                    simul_sample_nll, _, simul_sample_logps = self.net_lstm.run_eval(session, saved_sample_list)
                    model_sample_nll, _, model_sample_logps = self.eval_pi0(saved_sample_list)
                    _, _, model_sample_phis = self.eval_pi0(saved_sample_list, False)
                    kl_distance = simul_sample_nll - model_sample_nll

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                for name, v in lr_for_all.items():
                    info[name] = trfbase.to_str(v, '{:.1e}')
                info['lj_rate'] = self.lj_rate
                info['mv_rate'] = self.mv_rate
                info['train'] = np.mean(model_train_nll[-epoch_contain_step:])
                info['valid'] = model_valid_nll
                info['test'] = model_test_nll
                info['simu_valid'] = simul_valid_nll
                info['kl_dis'] = kl_distance
                trfbase.print_line(info)
                self.scalar_bank.write_summary(sv, session, 'nll_valid', model_valid_nll, step)
                self.scalar_bank.write_summary(sv, session, 'nll_test', model_test_nll, step)
                self.scalar_bank.write_summary(sv, session, 'kl_distance', kl_distance, step)
                self.summ_vars.write_summary(sv, session, step)

                ######################################
                # calculate the WER
                #####################################
                if epoch >= wer_next_epoch:
                    wer_next_epoch = int(epoch + wer_per_epoch)
                    if nbest is not None:
                        self.time_recoder.beg()
                        nbest.lmscore = -self.get_log_probs(nbest_list)
                        if lmscore_per_epoch is not None and int(epoch) % lmscore_per_epoch == 0:
                            wb.WriteScore(logname_lmscore + '.epoch{}.lmscore'.format(int(epoch)), nbest.lmscore)
                        else:
                            wb.WriteScore(logname_lmscore + '.lmscore', nbest.lmscore)
                        time_wer = self.time_recoder.end()
                        wer = nbest.wer()
                        print('wer={:.2f} lmscale={:.2f} score_time={:.2f}'.format(wer, nbest.lmscale, time_wer),
                              end=' ', flush=True)
                        self.scalar_bank.write_summary(sv, session, 'wer', wer, step)

                print('[end]')

                ########################################
                # more informations
                ########################################
                with self.time_recoder.recode('write_more_infos'):
                    f = self.write_files.get('dbg_info')
                    info = OrderedDict()
                    info['step'] = step
                    info['epoch'] = epoch
                    info['time'] = time_since_beg
                    info.update(lr_for_all)
                    info['lj_rate'] = self.lj_rate
                    info['lj_success'] = self.lj_success
                    info['lj_times'] = self.lj_times
                    info['mv_rate'] = self.mv_rate
                    info['mv_success'] = self.mv_success
                    info['mv_times'] = self.mv_times
                    info['model_train_nll'] = np.mean(model_train_nll[-epoch_contain_step:])
                    info['model_valid_nll'] = model_valid_nll
                    info['model_valid_nll_pi0'] = model_valid_nll_pi0
                    info['simul_valid_nll'] = simul_valid_nll
                    info['model_test_nll'] = model_test_nll
                    info['KL_distance'] = kl_distance
                    info['KL_distance_simulater_nll'] = simul_sample_nll
                    info['KL_distance_model_nll'] = model_sample_nll
                    info['vars_norm'] = session.run(self.param_norm)

                    # write pi
                    def atos(a):
                        return ' '.join(['{:>10.4f}'.format(p) for p in a[self.config.min_len:]])
                    info['[cur_pi]'] = atos(self.sample_cur_pi)
                    info['[all_pi]'] = atos(self.sample_acc_count/self.sample_acc_count.sum())
                    info['[pi_0__]'] = atos(self.config.pi_0)
                    info['[zeta__]'] = atos(self.zeta)
                    info['[logz__]'] = atos(self.logz)
                    for key, v in info.items():
                        f.write(key + '=' + trfbase.to_str(v) + '\n')
                    f.write('\n')
                    f.flush()

                # write samples
                f = self.write_files.get('sample')
                f.write('step={}\n'.format(step))
                f.write('[{:>4}]  [{:<10}]  [{:<10}]  [{:<10}]  [{}]\n'.format('id',
                                                                               'simul_logp',
                                                                               'model_logp', 'model_phi',
                                                                               'samples'))
                for i, (p1, p2, phi, s) in enumerate(zip(simul_sample_logps, model_sample_logps, model_sample_phis, saved_sample_list)):
                    f.write('[{:>4}]  {:<12.3f}  {:<12.3f}  {:<12.5f}  '.format(i, p1, p2, phi))
                    f.write(' '.join([str(w) for w in s]) + '\n')
                f.flush()
                saved_sample_list = []

                # write time info
                f = self.write_files.get('time')
                f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, (time.time() - time_beginning)/60))
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.items()]) + ' ')
                f.write('\n')
                f.flush()

                # write logps on valid set
                model_valid_nll_pi0, _, model_logps = self.eval_pi0(valid_list)
                _, _, model_phis = self.eval_pi0(valid_list, False)
                simul_valid_nll, _, simul_valid_logps = self.net_lstm.run_eval(session, valid_list)
                f = open(os.path.join(self.logdir, self.name + '.valid_probs'), 'wt')
                f.write('step={}\n'.format(step))
                f.write('model_valid_nll_pit={}\n'.format(model_valid_nll))
                f.write('model_valid_nll_pi0={}\n'.format(model_valid_nll_pi0))
                f.write('simul_valid_nll={}\n'.format(simul_valid_nll))
                f.write('[{:>4}]  [{:<10}]  [{:<10}]  [{:<10}]  [{}]\n'.format('id',
                                                                               'simul_logp',
                                                                               'model_logp', 'model_phi',
                                                                               'samples'))
                for i, (p1, p2, phi, s) in enumerate(zip(simul_valid_logps, model_logps, model_phis, valid_list)):
                    f.write('[{:>4}]  {:<12.3f}  {:<12.3f}  {:<12.5f}  '.format(i, p1, p2, phi))
                    f.write(' '.join([str(w) for w in s]) + '\n')
                f.flush()
                f.close()

                # write varis
                f = self.write_files.get('vars')
                f.write('step={}\n'.format(step))
                for v in self.net_trf.vars:
                    f.write(v.name + '=\n')
                    wb.write_array(f, session.run(v), '%+8.5f')
                f.flush()
                f.write('\n')

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



