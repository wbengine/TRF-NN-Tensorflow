import tensorflow as tf
from copy import deepcopy
import numpy as np
import time
import json
import tqdm

from base import *
from lm import loss
from lm import stop


class Config(wb.Config):
    def __init__(self, data=None):
        self.vocab_size = data.get_vocab_size() if data is not None else 10000
        self.batch_size = 20
        self.step_size = 20
        self.embedding_size = 200
        self.hidden_layers = 2
        self.hidden_size = 200
        self.softmax_type = 'Softmax'  # can be 'Softmax' or 'AdaptiveSoftmax' or 'Shortlist'
        self.adaptive_softmax_cutoff = [2000, self.vocab_size]  # for AdaptiveSoftmax, a list with the last value is vocab_size
        self.adaptive_softmax_project_factor = 4.  # for AdaptiveSoftmax a float >= 1.0
        self.softmax_pred_token_per_cluster = int(np.sqrt(self.vocab_size))  # used to accelerate the sampling, suggested as int(np.sqrt(vocab_size))
        self.fixed_logz_for_nce = 0  # the fixed logZ for NCE training
        self.init_weight = 0.1
        self.optimize_method = 'SGD'  # can be one of 'SGD', 'adam'
        self.max_grad_norm = 5.0      # grad_norm, can be None
        self.dropout = 0.0
        self.learning_rate = 1.  # [learning rate] initial values
        self.lr_decay = 0.5         # [learning rate] decay rate
        self.lr_decay_when = 4      # [learning rate] decay at epoch
        self.epoch_num = 13      # total epoch number
        self.early_stop = False
        self.save_embedding_path = None  # if not None, then save the embedding to file
        self.write_dbgs = None

    def __str__(self):
        s = 'lstm_e{}_h{}x{}'.format(self.embedding_size, self.hidden_size, self.hidden_layers)
        if self.softmax_type != 'Softmax':
            s += '_' + self.softmax_type
        if self.optimize_method != 'SGD':
            s += '_' + self.optimize_method
        return s


class Net(object):
    def __init__(self, config, data, is_training, name='lstm_net', reuse=None, static_rnn=False):
        self.is_training = is_training
        self.config = config
        self.state = None  # save the current state of LSMT
        self.data = data

        hidden_size = config.hidden_size
        hidden_layers = config.hidden_layers
        batch_size = config.batch_size
        dropout_prob = config.dropout
        step_num = None

        # a initializer for variables
        initializer = tf.random_uniform_initializer(-config.init_weight, config.init_weight)
        with tf.variable_scope(name, reuse=reuse, initializer=initializer):

            # Create LSTM cell
            def one_lstm_cell():
                c = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0., reuse=reuse)
                if self.is_training and dropout_prob > 0:
                    c = tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=1. - dropout_prob)
                return c

            cell = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(hidden_layers)])

            self._inputs = tf.placeholder(tf.int32, [batch_size, step_num], name='net_inputs')
            self._targets = tf.placeholder(tf.int32, [batch_size, step_num], name='net_targets')
            self._lengths = tf.ones([batch_size], dtype=tf.int32) * tf.shape(self._inputs)[1]
            self._initial_state = cell.zero_state(batch_size, tf.float32)  # initial state

            # bulid network...
            self.softmax, self._final_state, self._hidden_output = self.output(cell,
                                                                               self._initial_state, self._inputs,
                                                                               self._targets,
                                                                               self._lengths)

            # loss for each position
            # self._loss = self.softmax.loss
            # cost used to train function, or the negative-log-likelihood averaged over batches
            self._cost = tf.reduce_sum(self.softmax.loss)  # average over batches
            # all the trainalbe vairalbes
            self._variables = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

            if is_training:
                # as _lr and _global_step is not defined in variables_scope(),
                # as the default collection is tf.GraphKeys.TRAINABLE_VARIABLES
                # It should be initialized by session.run(tf.global_variables_initializer()) or
                # use tf.Train.Supervisor()
                self._lr = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')  # learning rate
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                tvars = self._variables
                self.grads = tf.gradients(self._cost, tvars)
                if config.max_grad_norm is not None:
                    self.grads, _ = tf.clip_by_global_norm(self.grads, config.max_grad_norm)
                if config.optimize_method.lower() == 'adam':
                    optimizer = tf.train.AdamOptimizer(self._lr)
                elif config.optimize_method.lower() == 'adagrad':
                    optimizer = tf.train.AdagradOptimizer(self._lr)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(self._lr)
                self._train_op = optimizer.apply_gradients(zip(self.grads, tvars), global_step=self.global_step)

                # update learining rate
                self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
                self._update_lr = tf.assign(self._lr, self._new_lr)

            else:
                self._train_op = tf.no_op()
                self._update_lr = tf.no_op()

    def output(self, cell, _initial_state, _inputs, _targets=None, _lengths=None):
        # word embedding
        word_embedding = tf.get_variable('word_embedding',
                                         [self.config.vocab_size, self.config.embedding_size],
                                         dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(word_embedding, _inputs)

        # dropout
        if self.is_training and self.config.dropout > 0:
            inputs = tf.nn.dropout(inputs, keep_prob=1. - self.config.dropout)

        # lstm
        # using tf.nn.dynamic_rnn to remove the constrains of step size.
        outputs, state = tf.nn.dynamic_rnn(cell,
                                           inputs=inputs,
                                           initial_state=_initial_state,
                                           sequence_length=_lengths)

        # softmax
        if self.config.softmax_type == 'AdaptiveSoftmax':
            print('apply adaptiveSoftmax')
            softmax = layers.AdaptiveSoftmax(outputs, _targets,
                                             cutoff=self.config.adaptive_softmax_cutoff,
                                             project_factor=self.config.adaptive_softmax_project_factor,
                                             name='AdaptiveSoftmax')
        elif self.config.softmax_type == 'Softmax':
            softmax = layers.Softmax(outputs, _targets, self.config.vocab_size,
                                     self.config.softmax_pred_token_per_cluster,
                                     name='Softmax')
        elif self.config.softmax_type == 'Shortlist':
            softmax = layers.ShortlistSoftmax(outputs, _targets,
                                              shortlist=self.config.adaptive_softmax_cutoff,
                                              stride=self.config.softmax_pred_token_per_cluster,
                                              name='Shortlist')
        elif self.config.softmax_type == 'BNCE':
            if self.data is None:
                noise_probs = np.ones(self.config.vocab_size) / self.config.vocab_size
                print('[lstmlm.Net] warning, using the uniform noise_probs!!!!')
            else:
                noise_probs = self.data.get_unigram()
            softmax = loss.BNCELoss(outputs, _targets, self.config.vocab_size,
                                    noise_probs=noise_probs,
                                    logZ=self.config.fixed_logz_for_nce,
                                    name='BNCELoss')
        else:
            raise TypeError('Unknown softmax method ', self.config.softmax_type)

        return softmax, state, outputs

    @property
    def hidden_output(self):
        return self._hidden_output

    @property
    def variables(self):
        return self._variables

    @property
    def cost(self):
        # the negative-log-likelihood averaged over batches
        return self._cost

    @property
    def draw(self):
        return self.softmax.draw

    @property
    def logps(self):
        return self.softmax.logps

    @property
    def dbgs(self):
        return self.softmax.dbgs

    def set_zero_state(self, session):
        """set the lstm state to zeros"""
        self.state = session.run(self._initial_state)

    def set_lr(self, session, learning_rate):
        """set the learning rate"""
        session.run(self._update_lr, {self._new_lr: learning_rate})

    def feed_state(self, state):
        """
        return feed dictionary used in session.run,
        to set the initial state

        Args:
            state: the lstm state, a tuple of tuples

        Returns:
            a feed dict
        """
        feed_dict = {}
        for (_c, _h), (c, h) in zip(self._initial_state, state):
            feed_dict[_c] = c
            feed_dict[_h] = h
        return feed_dict

    def run(self, session, x, y, ops=None):
        """
        given the inputs x, targets y, initial LSTM state,
        run operations.

        Args:
            session: tf.Session()
            x: inputs, of shape (batch_size, step_size), tf.int32
            y: targets, of shape (batch_size, step_size), tf.int32
            n: lengths, of shape (batch_size)
            ops: a list of operations,
                such as self._cost, self._loss, self._train_op
                if None, default is [self._cost, self._train_op]
                which can be used to training the model.
                There are not need to specifiy the self._final_state and
                self._summary_op operation, as they are add automaticly

        Returns:
            the return of ops except:
                self._final_state ( saved in self.state )
                self._summary_ops ( saved in self.summ)
                self._train_op ( donot return the value )

        """
        if self.state is None:
            self.set_zero_state(session)

        feed_dict = {self._inputs: x, self._targets: y}
        feed_dict.update(self.feed_state(self.state))  # feed state

        if ops is None:
            ops = [self._cost, self._train_op]
        else:
            ops = list(ops)  # copy the list
        ops.append(self._final_state)

        res = session.run(ops, feed_dict)
        # the final_state and summary op are saved in member variables
        self.state = res[-1]
        res = res[0: -1]

        # remove the train-op
        if self._train_op in ops:
            del res[ops.index(self._train_op)]

        if len(res) == 1:
            return res[0]
        return tuple(res)

    def run_predict(self, session, x, ops=None):
        """
        given the inputs x, initial LSTM state,
        calculate the predict prob of next words

        Args:
            session: tf.Session()
            x: inputs, of shape (batch_size, seq_len)
            ops: a list of operations,
                if ops==None, the default opts is only contains [self._final_state]

        Returns:
            the retured values of ops except self._final_state

        """
        if self.state is None:
            self.set_zero_state(session)  # get the zero state of current LSMT

        feed_dict = {self._inputs: x}
        feed_dict.update(self.feed_state(self.state))

        if ops is None:
            ops = []
        else:
            ops = list(ops)
        ops.append(self._final_state)

        res = session.run(ops, feed_dict)
        self.state = res[-1]
        res = res[0: -1]

        # for one value to return
        if len(res) == 1:
            return res[0]
        return tuple(res)


class LM(object):
    def __init__(self, config, data=None, device='/gpu:0', name='lstmlm', default_path=None):
        """
        Create a lstm LM. There are two networks which share the variables:
        One for training with batch size > 1,
        One for evaulation with batch size = 1

        Args:
            config: lstm configuration
            device: device string
            name: name of current lstm, used in variable_scope
            default_path: save to / restore from the files
        """
        self.name = name
        self.config = config
        self.data = data
        self.default_path = default_path
        with tf.device(device), tf.name_scope('Train'):
            # use for training
            # containing a train_op operation
            self.train_net = Net(config, data, is_training=True, name=name, reuse=None)
        with tf.device(device), tf.name_scope('Valid'):
            # use for evaluate with batch computation
            self.valid_net = Net(config, data, is_training=False, name=name, reuse=True)
        with tf.device(device), tf.name_scope('Eval'):
            # use for evaluate with batch=1
            eval_config = deepcopy(config)
            eval_config.batch_size = 1  # set batch_size to 1
            self.eval_net = Net(eval_config, data, is_training=False, name=name, reuse=True)

        self.param_num = tf.add_n([tf.size(v) for v in tf.trainable_variables()])
        self.saver = tf.train.Saver(self.train_net.variables)
        self.config = config

    def set_lr(self, session, lr):
        """set the learning rate"""
        self.train_net.set_lr(session, lr)

    def update(self, session, x, y):
        """
        update variables

        Args:
            session: tf.session
            x: inputs, of shape  (config.batch_size, config.step_size)
            y: targets, of shape (config.batch_size, config.step_size)

        Returns:
            tuple (
                cost: cross-entropy averaged over batch size,
                summary: the computed summary used to output
            }
        """
        cost, logps = self.train_net.run(session, x, y,
                                         [self.train_net.cost, self.train_net.logps,
                                          self.train_net._train_op])

        if self.config.write_dbgs is not None:
            with open(self.config.write_dbgs, 'a') as f:
                dbgs = self.train_net.run(session, x, y, [self.train_net.dbgs])
                f.write('step=%d\n' % session.run(self.train_net.global_step))
                np.set_printoptions(threshold=2000)
                for key, v in dbgs.items():
                    s = np.array2string(v, formatter={'float_kind': lambda x: "%.4f" % x})
                    f.write(key + '=' + s + '\n')
                f.flush()

        return cost, logps

    def sequence_update(self, session, seq_list):
        """

        Args:
            session: tf.session
            seq_list: a list of sequences

        Returns:

        """
        max_len = np.max([len(x) for x in seq_list])
        num = len(seq_list)
        input_x = np.zeros((num, max_len), dtype='int32')
        input_n = np.array([len(x) for x in seq_list], dtype='int32')
        for i in range(num):
            input_x[i][0: input_n[i]] = np.array(seq_list[i])

        cost = self.train_net.run(session, input_x[:, 0: -1], input_x[:, 1:], input_n-1)
        return cost

    def save_emb(self, session):
        """
        Write the embedding to files

        Args:
            session: tf.Session()

        Returns:
            None
        """
        if self.train_net.config.save_embedding_path is not None:
            self.train_net.saver.save(session, self.train_net.config.save_embedding_path)

    def save(self, session, path=None):
        """
        Args:
            session: tf.Session()
            path: *.ckpt
        """
        if path is None:
            path = self.default_path

        self.saver.save(session, path)

        # save the config
        if path.rfind('.ckpt') != -1:
            path = path[0: path.rfind('.ckpt')]
        with open(path + '.config', 'wt') as f:
            f.write(self.name + '\n')
            self.config.save(f)

    def restore(self, session, path=None):
        if path is None:
            path = self.default_path

        self.saver.restore(session, path)

    @staticmethod
    def load(path, device='/gpu:0'):
        if path.rfind('.ckpt') != -1:
            path_name = path[0: path.rfind('.ckpt')]
        else:
            path_name = path

        with open(path_name + '.config', 'rt') as f:
            name = f.readline().split()[0]
            config = wb.Config.load(f)
            m = LM(config, device=device, name=name, default_path=path)
        return m

    def eval(self, session, raw_data, net=None, reset_state=True):
        """
        eval the ppl and nll of given raw data

        Args:
            raw_data: a list of id-lists
            session: tensorflow session
            net: indicate the network used to evaulate the PPL,
                can be one of self.valid_net (fast, default), self.eval_net (slow but precise)
            reset_state: if True (default), then zero the state

        Returns:
            NLL, ppl
        """

        if net is None:
            net = self.valid_net

        x_list, y_list = reader.produce_data_for_rnn(raw_data,
                                                     net.config.batch_size,
                                                     net.config.step_size,
                                                     include_residual_data=True)
        if reset_state:
            net.set_zero_state(session)
        total_cost = 0
        total_word = 0
        for x, y in zip(x_list, y_list):
            logps = net.run(session, x, y, [net.logps])
            total_cost += np.sum(-logps)  # as cost is averaged over batch
            total_word += np.size(x)

        seq_num = len(raw_data)
        nll = total_cost / seq_num
        ppl = np.exp(total_cost / total_word)
        return nll, ppl

    def rescore(self, session, seq_list, reset_state_for_sentence=True, pad_end_token_to_head=True):
        """
        compute the logprob of each sequence separately

        Args:
            session: tf.Session()
            seq_list: a list of sequence
            reset_state_for_sentence: if True, then reset the state for each sentences

        Returns:
            an np.array of shape [len(seq_list),],
            indicating the logprob of each sequence
        """

        # rm the beg-token of each sequences
        if self.data is not None:
            for s in seq_list:
                if s[0] == self.data.get_beg_token():
                    del s[0]

        if reset_state_for_sentence:
            # batch_parallized the computation
            inputs, lengths = reader.produce_data_to_trf(seq_list)
            if pad_end_token_to_head:
                # app the </s> to the head of each sentences
                inputs = np.pad(inputs, [[0, 0], [1, 0]], mode='constant', constant_values=seq_list[0][-1])
                lengths += 1
            return -self.conditional(session, inputs, 1, lengths, initial_state=True)

        else:
            net = self.eval_net
            net.set_zero_state(session)

            score = np.zeros(len(seq_list))
            for i, seq in tqdm.tqdm(enumerate(seq_list), total=len(seq_list)):
                score[i] = self.eval(session, [[seq[-1]] + seq], net=net,
                                     reset_state=reset_state_for_sentence)[0]
            return score

    def global_step(self):
        return self.train_net.global_step

    def train_batch_size(self):
        return self.config.batch_size

    def simulate(self, session, initial_seqs, sample_nums, initial_state=False, context_size=None):
        """
        given the initial_sequences, drow the following elements

        Args:
            session: tf.Session()
            initial_seqs: np.array, of shape (batch_size, sequence_len) of int32
            sample_nums: int or np.array, should >= 1
            initial_state: if True, then set the initial state to Zeros
            context_size: is not None and >=1 then only consinder the previous context_size values

        Returns:
            the final sequences, of shape (batch_size, sequence_len + np.max(sample_nums))
            the conditional logprob, of shape (batch_size)
        """
        net = self.valid_net
        input_batch = len(initial_seqs)
        if isinstance(sample_nums, int):
            sample_nums = [sample_nums] * input_batch
        sample_nums = np.array(sample_nums)

        # if the initial_seqs is []
        initial_logp = np.zeros(input_batch)
        if initial_seqs.shape[1] == 0:
            initial_seqs = np.random.randint(net.config.vocab_size, size=(input_batch, 1))
            initial_logp = np.ones(input_batch) * np.log(1.0 / net.config.vocab_size)
            sample_nums -= 1
            if np.all(sample_nums == 0):
                return initial_seqs, initial_logp

        max_sample_nums = np.max(sample_nums)

        final_seq_list = []
        final_logp_list = []
        for batch_beg in range(0, initial_seqs.shape[0], net.config.batch_size):
            seqs = initial_seqs[batch_beg: batch_beg + net.config.batch_size]
            pnum = net.config.batch_size - seqs.shape[0]
            seqs = np.pad(seqs, [[0, pnum], [0, 0]], 'edge')

            if initial_state:
                net.set_zero_state(session)

            # forward, to update states
            if context_size is not None and context_size >= 1:
                net.run_predict(session, seqs[:, -context_size:-1])
            else:
                net.run_predict(session, seqs[:, 0:-1])

            # sample
            logps = np.zeros((net.config.batch_size, 0))
            for i in range(max_sample_nums):
                # y:      (batch_size, step_size=1)
                # y_logp: (batch_size, step_size=1)
                y, y_logp = net.run_predict(session, seqs[:, -1:], net.softmax.draw)

                seqs = np.concatenate([seqs, y], axis=-1)
                logps = np.concatenate([logps, y_logp], axis=-1)

            final_seq_list.append(seqs)
            final_logp_list.append(logps)

        final_seqs = np.concatenate(final_seq_list, axis=0)[0: input_batch]
        final_logps = np.concatenate(final_logp_list, axis=0)[0: input_batch]

        final_logp = np.array([np.sum(a[0: n]) for a, n in zip(final_logps, sample_nums)])
        final_logp += initial_logp

        return final_seqs, final_logp

    def conditional(self, session, input_seqs, pos_vec, len_vec=None, initial_state=True, context_size=None):
        """
        calculate the conditional probabilities p(x[pos:] | x[0: pos])

        Args:
            session: tf.Session()
            input_seqs: input sequence, of shape (batch_size, sequence_len)
            pos_vec: integer/1-d vector, denoting the position of each sequenc
            len_vec: integer/1-d vector, denoting the length of each sequence.
                if None, then the length is input_seqs.shape[1]
            initial_state: if True, then set the initial state to Zeros
            context_size: is not None and >=1 then only consinder the previous context_size values

        Returns:
            1-d np.array of shape (batch_size),
            denoting the conditional log-probabilities

        """
        net = self.valid_net

        input_batch = len(input_seqs)
        if isinstance(pos_vec, int):
            pos_vec = np.array([pos_vec] * input_batch)
        if len_vec is None:
            len_vec = np.array([input_seqs.shape[1]] * input_batch)
        elif isinstance(len_vec, int):
            len_vec = np.array([len_vec] * input_batch)

        assert np.all(len_vec >= pos_vec)
        assert np.all(len_vec <= input_seqs.shape[1])

        # to make sure the pos_vec > 0
        initial_logps = np.zeros(input_batch)
        for i in range(input_batch):
            if pos_vec[i] == 0:
                initial_logps[i] = np.log(1.0 / net.config.vocab_size)
                pos_vec[i] += 1

        final_list = []
        for batch_beg in range(0, input_seqs.shape[0], net.config.batch_size):
            seqs = input_seqs[batch_beg: batch_beg + net.config.batch_size]
            pnum = net.config.batch_size - seqs.shape[0]
            seqs = np.pad(seqs, [[0, pnum], [0, 0]], 'edge')
            local_pos = pos_vec[batch_beg: batch_beg + net.config.batch_size]
            local_len = len_vec[batch_beg: batch_beg + net.config.batch_size]

            if initial_state:
                net.set_zero_state(session)

            if context_size is not None and context_size >= 1:
                new_seqs = []
                new_pos = []
                new_len = []
                for s, pos, l in zip(seqs, local_pos, local_len):
                    if pos > context_size:
                        cut_num = pos - context_size
                    else:
                        cut_num = 0
                    new_seqs.append(s[cut_num:])
                    new_pos.append(pos - cut_num)
                    new_len.append(l - cut_num)
                seqs, _ = reader.produce_data_to_trf(new_seqs)
                seqs = np.pad(seqs, [[0, pnum], [0, 0]], 'edge')
                local_pos = np.array(new_pos)
                local_len = np.array(new_len)

            # run predict
            maxlen = np.max(local_len)
            logps = net.run(session, seqs[:, 0:maxlen-1], seqs[:, 1:maxlen], [net.logps])  # the returned is the -logp

            cond_logps = [logp[k-1: n-1].sum() for logp, k, n in zip(logps, local_pos, local_len)]
            final_list.append(cond_logps)

        return np.concatenate(final_list, axis=0)[0: input_batch] + initial_logps

    def train(self, session, data, write_model,
              write_to_res=None, is_shuffle=True,
              print_per_epoch=0.1):

        if isinstance(data, reader.LargeData):
            self.train_for_large_data(session, data, write_model, write_to_res, is_shuffle, print_per_epoch)
        else:
            self.train_for_data(session, data, write_model, write_to_res, is_shuffle, print_per_epoch)

    def train_for_data(self, session, data, write_model,
                       write_to_res=None, is_shuffle=True,
                       print_per_epoch=0.1):
        """
        training the lstm LM
        :param session: tf.session()
        :param data:  the reader.data() instance
        :param write_model: str, write model
        :param write_to_res: tuple (res_file_name, write_model_name)
        :return:
        """

        print('param_num={:,}'.format(session.run(self.param_num)))

        print('prepare data, beg_token={} end_token={}'.format(data.beg_token_str, data.end_token_str))
        if data.get_beg_token() is None:
            print('data valid: no beg token')
        else:
            print('data contains beg-tokens, rm the beg-tokens in all data sets')
            data.rm_beg_tokens_in_datas()
            print('result data:')
            print(data.datas[0][10])
            print(data.datas[1][10])
            print(data.datas[2][10])

        if not wb.exists(write_model + '.index'):
            time_beg = time.time()
            train_cost = []
            train_nll = []
            total_word = 0
            early_stop = stop.EarlyStop(self.config.learning_rate, 1e-3, delay_min_rate=0.05)

            for epoch in range(self.config.epoch_num):
                if is_shuffle:
                    np.random.shuffle(data.datas[0])
                x_lists, y_lists = reader.produce_data_for_rnn(data.datas[0], self.train_batch_size(), self.config.step_size)
                epoch_size = len(x_lists)

                # update learning rate
                if self.config.early_stop:
                    lr = early_stop.lr
                else:
                    lr = self.config.learning_rate * self.config.lr_decay ** max(epoch + 1 - self.config.lr_decay_when, 0.0)
                self.set_lr(session, lr)

                # run epoch
                for i, (x, y) in enumerate(zip(x_lists, y_lists)):

                    cost, logps = self.update(session, x, y)
                    train_cost.append(cost)
                    train_nll.append(-np.mean(logps))
                    total_word += np.size(x)

                    if print_per_epoch == 0 or (i + 1) % (int(epoch_size * print_per_epoch)) == 0:
                        time_since_beg = time.time() - time_beg
                        print('epoch={:.3f} w/s={:.2f} cost={:.3f} train_ppl={:.3f} '
                              'lr={:.3f} time_since_begin={:.2f}m'.format(
                            epoch + (i + 1) / epoch_size,
                            total_word / time_since_beg,
                            np.mean(train_cost[-epoch_size:]),
                            np.exp(np.mean(train_nll[-epoch_size:])),
                            lr,
                            time_since_beg / 60
                        ))
                        # lm.save_emb(session)
                        # summ_var.write_summary(sv, session)

                eval_valid = self.eval(session, data.datas[1], net=self.valid_net)
                # eval_test = self.eval(session, data.datas[2])

                # write ppl to log
                print('epoch={:d} ppl-valid={:.3f}'.format(epoch + 1, eval_valid[1]))
                # summ_bank.write_summary(sv, session, 'ppl_valid', eval_valid[1])
                # summ_bank.write_summary(sv, session, 'ppl_test', eval_test[1])

                if self.config.early_stop:
                    if early_stop.verify(eval_valid[1]) is None:
                        print('early stopped !')
                        break

            self.save(session, write_model)
        else:
            print('restore the exist model files:', write_model)
            self.restore(session, write_model)

        self.save(session, write_model)
        print('compute the ppls...')
        nll_train, ppl_train = self.eval(session, data.datas[0])
        nll_valid, ppl_valid = self.eval(session, data.datas[1])
        nll_test, ppl_test = self.eval(session, data.datas[2], net=self.eval_net)
        print('ppl-train={:.3f} ppl-valid={:.3f} ppl-test={:.3f}'.format(ppl_train, ppl_valid, ppl_test))

        if write_to_res is not None:
            res = wb.FRes(write_to_res[0])
            res.AddLLPPL(write_to_res[1],
                         [nll_train, nll_valid, nll_train],
                         [ppl_train, ppl_valid, ppl_test])

    def train_for_large_data(self, session, data, write_model,
                             write_to_res=None, is_shuffle=True,
                             print_per_epoch=0.1):

        print('param_num={:,}'.format(session.run(self.param_num)))

        if not wb.exists(write_model + '.index'):
            time_beg = time.time()
            train_cost = []
            train_nll = []
            total_word = 0

            for epoch in range(self.config.epoch_num):

                # update learning rate
                lr = self.config.learning_rate * self.config.lr_decay ** max(epoch + 1 - self.config.lr_decay_when, 0.0)
                self.set_lr(session, lr)

                # run epoch
                for file_num in range(len(data.train_file_list)):
                    if is_shuffle:
                        np.random.shuffle(data.datas[0])

                    x_lists, y_lists = reader.produce_data_for_rnn(data.datas[0],
                                                                   self.train_batch_size(),
                                                                   self.config.step_size)
                    epoch_size = len(x_lists)

                    for i, (x, y) in enumerate(zip(x_lists, y_lists)):

                        # update parameters
                        cost, logps = self.update(session, x, y)
                        train_cost.append(cost)
                        train_nll.append(-np.mean(logps))
                        total_word += np.size(x)

                        if print_per_epoch == 0 or (i + 1) % int(epoch_size * print_per_epoch) == 0:
                            time_since_beg = time.time() - time_beg
                            print('epoch={:.3f} w/s={:.2f} cost={:.3f} train_ppl={:.3f} '
                                  'lr={:.3f} time_since_begin={:.2f}m'.format(
                                epoch + (file_num + (i + 1) / epoch_size) / len(data.train_file_list),
                                total_word / time_since_beg,
                                np.mean(train_cost[-epoch_size:]),
                                np.exp(np.mean(train_nll[-epoch_size:])),
                                lr,
                                time_since_beg / 60
                            ))

                    eval_valid = self.eval(session, data.datas[1], net=self.valid_net)
                    eval_test = self.eval(session, data.datas[2], net=self.valid_net)

                    # write ppl to log
                    print('ppl-valid={:.3f} ppl-test={:.3f}'.format(
                        eval_valid[1], eval_test[1]))
                    self.save(session, write_model)

                    data.next_train()

            self.save(session, write_model)
        else:
            print('restore the exist model files:', write_model)
            self.restore(session, write_model)

        ##################################################
        # evaulation
        ##################################################
        self.save(session, write_model)
        print('compute the ppls...')
        nll_train, ppl_train = self.eval(session, data.datas[0])
        nll_valid, ppl_valid = self.eval(session, data.datas[1])
        nll_test, ppl_test = self.eval(session, data.datas[2], net=self.eval_net)
        print('ppl-train={:.3f} ppl-valid={:.3f} ppl-test={:.3f}'.format(ppl_train, ppl_valid, ppl_test))

        if write_to_res is not None:
            res = wb.FRes(write_to_res[0])
            res.AddLLPPL(write_to_res[1],
                         [nll_train, nll_valid, nll_train],
                         [ppl_train, ppl_valid, ppl_test])


def load_config(path):
    """
    load the config of LSTMLM
    """
    if path.rfind('.ckpt') != -1:
        path_name = path[0: path.rfind('.ckpt')]
    else:
        path_name = path

    with open(path_name + '.config', 'rt') as f:
        name = f.readline().split()[0]
        config = wb.Config.load(f)
        return config


def load(path, device='/gpu:0'):
    """
    load the config and create a lsmtlm,
    please call LM.restore to restore the variables in ckpt before using this model
    Usage:
        m = lstmlm.load(path, device)      # create the network

        sess = tf.Session(...config...)    # create a session
        m.restroe(sess)                    # restore the variables

        # ... use the model ...
    """
    if path.rfind('.ckpt') != -1:
        path_name = path[0: path.rfind('.ckpt')]
    else:
        path_name = path

    with open(path_name + '.config', 'rt') as f:
        name = f.readline().split()[0]
        config = wb.Config.load(f)
        m = LM(config, device=device, name=name, default_path=path)
    return m


class DistributedNet(object):
    def __init__(self, config, is_training, device_list, name='lstm_net', reuse=None):

        self.distributed_num = len(device_list)
        self.config = config
        max_grad_norm = config.max_grad_norm
        config.max_grad_norm = None

        with tf.device(device_list[0]):
            self.net_list = []
            for i, dev in enumerate(device_list):
                with tf.device(dev), tf.name_scope('%s_%d' % (name, i)):
                    net = Net(config, is_training=is_training, name=name, reuse=reuse if i == 0 else True)
                    self.net_list.append(net)

            # total cost, avraged over batch_size
            self.cost = tf.add_n([net.cost for net in self.net_list]) / len(self.net_list)
            if is_training:
                tvars = self.net_list[0].variables
                # average all the gradients
                averaged_grads = []
                grads_list = [net.grads for net in self.net_list]
                for gs in zip(*grads_list):
                    # gs is like (g0_at_gpu0, g0_at_gpu1, ..., g0_at_gpuN), ...
                    grads = tf.concat([tf.expand_dims(g, axis=0) for g in gs], axis=0)
                    averaged_grads.append(tf.reduce_mean(grads, axis=0))

                # optimize the parameters
                self._lr = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')  # learning rate
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                if max_grad_norm is not None:
                    averaged_grads, _ = tf.clip_by_global_norm(averaged_grads, max_grad_norm)
                if config.optimize_method.lower() == 'adam':
                    optimizer = tf.train.AdamOptimizer(self._lr)
                elif config.optimize_method.lower() == 'adagrad':
                    optimizer = tf.train.AdagradOptimizer(self._lr)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(self._lr)
                self._train_op = optimizer.apply_gradients(zip(averaged_grads, tvars), global_step=self.global_step)

                # update learining rate
                self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
                self._update_lr = tf.assign(self._lr, self._new_lr)
            else:
                self._update_lr = tf.no_op()
                self._train_op = tf.no_op()

    def set_lr(self, session, lr):
        """set the learning rate"""
        session.run(self._update_lr, {self._new_lr: lr})

    def set_zero_state(self, session):
        for net in self.net_list:
            net.set_zero_state(session)

    def run(self, session, x, y, ops=None):
        """

        Args:
            session: tf.Session()
            x: the inputs
            y: the targets
            ops: list of operations, if None, then compute the cost and perform train_op

        Returns:

        """
        if len(x) != self.distributed_num * self.config.batch_size:
            raise TypeError('the batch_size of input data[%d] '
                            'is not equal to distributed_num[%d] '
                            '* model_batch_size[%d]' %
                            (len(x), self.distributed_num, self.config.batch_size))

        feed_dict = dict()
        # feed for each net
        for net, net_x, net_y, in zip(self.net_list,
                                      np.split(x, self.distributed_num),
                                      np.split(y, self.distributed_num)):
            if net.state is None:
                net.set_zero_state(session)
            # feed inputs and targets
            feed_dict[net._inputs] = net_x
            feed_dict[net._targets] = net_y
            # feed states
            feed_dict.update(net.feed_state(net.state))

        # outputs
        if ops is None:
            opeartions = {'cost': self.cost, 'train': self._train_op}
        else:
            if isinstance(ops, list):
                opeartions = dict([('op_%d' % i, ops[i]) for i in range(len(ops))])
            else:
                opeartions = {'op_0': ops}
        for i, net in enumerate(self.net_list):
            opeartions['final_state_%d' % i] = net._final_state

        # run
        res = session.run(opeartions, feed_dict)

        # process the results
        for i, net in enumerate(self.net_list):
            net.state = res['final_state_%d' % i]
            del res['final_state_%d' % i]
        if 'train' in res:
            del res['train']

        # return
        res = [x[1] for x in res.items()]
        if len(res) == 1:
            return res[0]
        return res

    def run_predict(self, session, x, ops=None):
        if len(x) != self.distributed_num * self.config.batch_size:
            raise TypeError('the batch_size of input data[%d] '
                            'is not equal to distributed_num[%d] '
                            '* model_batch_size[%d]' %
                            (len(x), self.distributed_num, self.config.batch_size))

        feed_dict = dict()
        # feed for each net
        for net, net_x, in zip(self.net_list, np.split(x, self.distributed_num)):
            if net.state is None:
                net.set_zero_state(session)
            # feed inputs and targets
            feed_dict[net._inputs] = net_x
            # feed states
            feed_dict.update(net.feed_state(net.state))

        # outputs
        if ops is None:
            opeartions = dict()
        else:
            if isinstance(ops, list):
                opeartions = dict([('op_%d' % i, ops[i]) for i in range(len(ops))])
            else:
                opeartions = {'op_0': ops}
        for i, net in enumerate(self.net_list):
            opeartions['final_state_%d' % i] = net._final_state

        # run
        res = session.run(opeartions, feed_dict)

        # process the results
        for i, net in enumerate(self.net_list):
            net.state = res['final_state_%d' % i]
            del res['final_state_%d' % i]

        # return
        res = [x[1] for x in res.items()]
        if len(res) == 1:
            return res[0]
        return res


class FastLM(object):
    def __init__(self, config, eval_config=None, device_list=['/gpu:0'], name='lstm-lm'):
        with tf.name_scope('Train'):
            self.train_net = DistributedNet(config, is_training=True,
                                            device_list=device_list, name=name, reuse=None)
        with tf.name_scope('Valid'):
            self.valid_net = DistributedNet(config, is_training=False,
                                            device_list=device_list, name=name, reuse=True)
        with tf.name_scope('Eval'):
            if eval_config is None:
                eval_config = deepcopy(config)
                eval_config.batch_size = 1  # set batch_size to 1
            self.eval_net = DistributedNet(eval_config, is_training=False,
                                           device_list=device_list[0:1], name=name, reuse=True)

        self.saver = tf.train.Saver()

    def set_lr(self, session, lr):
        """set the learning rate"""
        self.train_net.set_lr(session, lr)

    def update(self, session, x, y):
        cost = self.train_net.run(session, x, y)
        return cost

    def save(self, session, path):
        self.saver.save(session, path)

    def eval(self, session, raw_data, net=None, reset_state=True):
        if net is None:
            net = self.eval_net

        x_list, y_list = reader.produce_data_for_rnn(raw_data,
                                                     net.config.batch_size * net.distributed_num,
                                                     net.config.step_size,
                                                     include_residual_data=True)
        if reset_state:
            net.set_zero_state(session)
        total_cost = 0
        total_word = 0
        for x, y in zip(x_list, y_list):
            cost = net.run(session, x, y)
            total_cost += cost * net.config.batch_size * net.distributed_num  # as cost is averaged over batch
            total_word += np.size(x)

        seq_num = len(raw_data)
        nll = total_cost / seq_num
        ppl = np.exp(total_cost / total_word)
        return nll, ppl

    def rescore(self, session, seq_list, reset_state_for_sentence=False):
        net = self.eval_net
        net.set_zero_state(session)

        score = np.zeros(len(seq_list))
        for i, seq in tqdm.tqdm(enumerate(seq_list), total=len(seq_list)):
            score[i] = self.eval(session, [[seq[-1]] + seq], net=net,
                                 reset_state=reset_state_for_sentence)[0]
        return score

    def global_step(self):
        return self.train_net.global_step

    def train_batch_size(self):
        return self.train_net.config.batch_size * self.train_net.distributed_num


def compare_softmax(_):
    data = reader.Data().load_raw_data(reader.ptb_raw_dir())
    data.write_data(1, 'ptb.valid.id')
    data.write_vocab('ptb_vocab.txt')

    with tf.Graph().as_default():
        config = Config()
        config.vocab_size = data.get_vocab_size()

        config1 = deepcopy(config)
        config1.softmax_type = 'Softmax'
        lm_softmax = LM(config1, device='/gpu:0', name='lstm_softmax')
        config2 = deepcopy(config)
        config2.softmax_type = 'AdaptiveSoftmax'
        lm_adaptive = LM(config2, device='/gpu:0', name='lstm_adaptive')

        tf.summary.FileWriter('comp_softmax_logs', graph=tf.get_default_graph())

        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as session:
            session.run(tf.global_variables_initializer())

            print('run evaulation...')

            # time_beg = time.time()
            # print(lm_softmax.eval(session, data.datas[2]))
            # print('time_softmax=', time.time() - time_beg)
            #
            # time_beg = time.time()
            # print(lm_adaptive.eval(session, data.datas[2]))
            # print('time_adaptive=', time.time() - time_beg)

            net = lm_softmax.sample_net
            max_len = 100
            full_seq = np.random.randint(data.get_vocab_size(), size=(net.config.batch_size, max_len))

            # verified
            beg_len = 1
            full_seq, cond_logp = lm_softmax.simulate(session, full_seq[:, 0:beg_len], max_len-beg_len, initial_state=True)
            cond_logp2 = lm_softmax.conditional(session, full_seq, beg_len, max_len, initial_state=True)
            print('cond_logp1=', cond_logp)
            print('cond_logp2=', cond_logp2)

            time_beg = time.time()
            for t in range(10):
                net.set_zero_state(session)
                for i in range(full_seq.shape[1]-1):
                    net.run_predict(session, full_seq[:, i:i+1])
            print('pred=', time.time() - time_beg)

            time_beg = time.time()
            for t in range(10):
                lm_softmax.simulate(session, full_seq[:, 0:1], max_len-1, initial_state=True)
            print('simu=', time.time() - time_beg)

            time_beg = time.time()
            for t in range(10):
                net.set_zero_state(session)
                for i in range(full_seq.shape[1]-1):
                    net.run(session, full_seq[:, i:i+1], full_seq[:, i+1:i+2], [net.logps])
            print('loss=', time.time() - time_beg)


            # time_beg = time.time()
            # max_len = 100
            # for t in range(100):
            #     beg_len = 1
            #     beg_seq = np.random.randint(data.get_vocab_size(), size=(config.batch_size, beg_len))
            #     full_seq, cond_logp = lm_softmax.simulate(session, beg_seq, max_len-beg_len, initial_state=True)
            #
            #     # full_seq = np.random.randint(data.get_vocab_size(), size=(config.batch_size, max_len))
            #     # cond_logp2 = lm_softmax.conditional(session, full_seq,
            #     #                                     beg_len, max_len,
            #     #                                     initial_state=True)
            #
            #     # print(full_seq)
            #     # print(cond_logp)
            #     # print(cond_logp2)
            #
            # print('time=', time.time()-time_beg)


def main(_):
    # data = reader.Data().load_raw_data(reader.ptb_raw_dir())
    data = reader.Data().load_raw_data(reader.word_raw_dir(),
                                       add_beg_token='<s>', add_end_token='</s>',
                                       add_unknwon_token=None)
    data.write_vocab('ptb_vocab.txt')
    data.write_data(1, 'ptb.valid.id')

    config = Config()
    config.vocab_size = data.get_vocab_size()
    config.hidden_size = 128
    config.hidden_layers = 1
    # config.optimize_method = 'Adam'
    # config.learning_rate = 0.001
    config.save_embedding_path = './embedding/word_emb_{}x{}.ckpt'.format(config.vocab_size, config.hidden_size)
    config.epoch_num = 1

    x_lists, y_lists = reader.produce_data_for_rnn(data.datas[0], config.batch_size, config.step_size)

    with tf.Graph().as_default():
        lm = LM(config, device='/gpu:0', name='lstmlm')

        # used to write ppl on valid/test set
        with tf.name_scope('ppl'):
            _valid_ppl = tf.placeholder(dtype=tf.float32, shape=[], name='valid')
            _test_ppl = tf.placeholder(dtype=tf.float32, shape=[], name='test')
            tf.summary.scalar('ppl/valid', _valid_ppl, collections=['ppl'])
            tf.summary.scalar('ppl/test', _test_ppl, collections=['ppl'])
            summ_ppl = tf.summary.merge_all(key='ppl')

        sv = tf.train.Supervisor(logdir='logs', summary_op=None, global_step=lm.global_step())
        sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:

            time_beg = time.time()
            train_cost = []
            total_word = 0
            epoch_size = len(x_lists)

            for epoch in range(config.epoch_num):

                # update learning rate
                lr = config.learning_rate * config.lr_decay ** max(epoch + 1 - config.lr_decay_when, 0.0)
                lm.set_lr(session, lr)

                # run epoch
                for i, (x, y) in enumerate(zip(x_lists, y_lists)):

                    cost = lm.update(session, x, y)
                    train_cost.append(cost)
                    total_word += np.size(x)

                    sv.summary_computed(session, session.run(lm.train_net.summary))

                    if (i + 1) % (epoch_size//10) == 0:

                        time_since_beg = time.time() - time_beg
                        print('epoch={:.3f} w/s={:.2f} train_ppl={:.3f} valid_ppl={:.3f} lr={:.3f} time_since_begin={:.2f}m'.format(
                            epoch + (i+1)/epoch_size,
                            total_word / time_since_beg,
                            np.exp(np.mean(train_cost[-epoch_size:]) / config.batch_size),
                            lm.eval(session, data.datas[1])[1],
                            lr,
                            time_since_beg / 60
                        ))
                        lm.save_emb(session)

                eval_valid = lm.eval(session, data.datas[1])
                eval_test = lm.eval(session, data.datas[2])
                # write ppl to log
                sv.summary_computed(session,
                                    session.run(summ_ppl, {_valid_ppl: eval_valid[1], _test_ppl: eval_test[1]}),
                                    global_step=epoch)
                print('epoch={:d} ppl-valid={:.3f} ppl-test={:.3f}'.format(epoch+1, eval_valid[1], eval_test[1]))


if __name__ == '__main__':
    tf.app.run(main=main)
