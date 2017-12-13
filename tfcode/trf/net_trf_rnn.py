import tensorflow as tf
import numpy as np

from . import net_trf_cnn
from . import layers
from . import word2vec
from . import reader
from . import wblib as wb


class NetTRF(net_trf_cnn.NetTRF):
    class Config(wb.PPrintObj):
        def __init__(self, data):
            self.min_len = data.get_min_len()
            self.max_len = data.get_max_len()
            self.vocab_size = data.get_vocab_size()
            self.beg_token = data.get_beg_token()
            self.end_token = data.get_end_token()
            self.pi_true = data.get_pi_true()
            self.pi_0 = data.get_pi0(self.pi_true)

            # for structure
            self.type = 'lstm'  # of 'lstm' or 'rnn'
            self.embedding_dim = 200
            self.hidden_dim = 200
            self.hidden_layers = 2
            self.dropout = 0.0
            self.load_embedding_path = None

            # for training
            self.init_weight = 0.1
            self.opt_method = 'adam'
            self.train_batch_size = 100
            self.sample_batch_size = 100
            self.update_batch_size = 100  # used in run_training, to constraint memory usage
            self.zeta_gap = 10
            self.max_grad_norm = 10.0

        def __str__(self):
            return 'b{}_e{}_h{}x{}'.format(self.type, self.embedding_dim, self.hidden_dim, self.hidden_layers)

        def initial_zeta(self):
            len_num = self.max_len - self.min_len + 1
            logz = np.append(np.zeros(self.min_len),
                             np.log(self.vocab_size) * np.linspace(1, len_num, len_num))
            zeta = logz - logz[self.min_len]
            return zeta

    def __init__(self, config, reuse=None, is_training=True, name='net_trf',  propose_lstm=None):
        super().__init__(config, reuse=reuse, is_training=is_training, name=name)

        # for pre-train
        self.pre_train = self.get_pre_train(self._inputs, self._lengths, reuse=reuse)
        if reuse is None:
            print('pre_train extra-variables in %s' % name)
            for v in self.pre_train.tvars:
                if v not in self.vars:
                    print('\t{} {}'.format(v.name, v.shape))

        # copy pre-train variables
        if propose_lstm is not None:
            self.copy_ops = self.get_copy_variables_to_propose_lstm(propose_lstm.vars)
        else:
            self.copy_ops = None

    def compute_rnn(self, _inputs, _lengths, reuse=True):
        # LSTM Cells
        # lstm cell
        # Create LSTM cell
        def one_lstm_cell():
            if self.config.type == 'lstm':
                c = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, forget_bias=0., reuse=reuse)
            elif self.config.type == 'rnn':
                c = tf.contrib.rnn.BasicRNNCell(self.config.hidden_dim, activation=tf.nn.tanh, reuse=reuse)
            else:
                raise TypeError('undefined rnn type = ' + self.config.type)
            if self.config.dropout > 0:
                c = tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=1. - self.config.dropout)
            return c

        cell_fw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.hidden_layers)])
        cell_bw = tf.contrib.rnn.MultiRNNCell([one_lstm_cell() for _ in range(self.config.hidden_layers)])

        vocab_size = self.config.vocab_size
        embedding_dim = self.config.embedding_dim

        batch_size = tf.shape(_inputs)[0]

        # embedding layers
        with tf.device('/cpu:0'):
            if self.config.load_embedding_path is not None:
                print('read init embedding vector from', self.config.load_embedding_path)
                emb_init_value = word2vec.read_vec(self.config.load_embedding_path)
                if emb_init_value.shape != (vocab_size, embedding_dim):
                    raise TypeError('the reading embedding with shape ' +
                                    str(emb_init_value.shape) +
                                    ' does not match current shape ' +
                                    str([vocab_size, embedding_dim]) +
                                    '\nform path ' + self.config.load_embedding_path)
                word_embedding = tf.get_variable('word_embedding',
                                                 [vocab_size, embedding_dim], dtype=tf.float32,
                                                 initializer=tf.constant_initializer(emb_init_value),
                                                 trainable=True)
            else:
                word_embedding = tf.get_variable('word_embedding',
                                                 [vocab_size, embedding_dim], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(word_embedding, _inputs)  # (batch_size, seq_len, emb_dim)

        # dropout
        if self.config.dropout > 0:
            inputs = tf.nn.dropout(inputs, keep_prob=1. - self.config.dropout)

        # lstm
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                          inputs=inputs,
                                                          sequence_length=_lengths,
                                                          initial_state_fw=cell_fw.zero_state(batch_size, tf.float32),
                                                          initial_state_bw=cell_fw.zero_state(batch_size, tf.float32))

        return outputs[0], outputs[1], states

    def compute_phi(self, _inputs, _lengths, reuse=True):
        """reuse all the variables to compute the phi"""

        outputs_fw, outputs_bw, _ = self.compute_rnn(_inputs, _lengths, reuse)

        inputs = tf.concat([outputs_fw, outputs_bw], axis=2)

        # final layers
        batch_size = tf.shape(_inputs)[0]
        len_mask = tf.sequence_mask(_lengths, maxlen=tf.shape(_inputs)[1], dtype=tf.float32)
        outputs = layers.linear(inputs, 1, name='final_linear')  # [batch_size, max_len, 1]
        outputs = tf.reshape(outputs, [batch_size, -1])  # [batch_size, max_len]
        outputs = outputs * len_mask
        outputs = tf.reduce_sum(outputs, axis=-1)  # of shape [batch_size]

        return outputs

    def get_pre_train(self, _inputs, _lengths, reuse=True):
        # the b-LSTM is already created
        with tf.variable_scope(self.name, reuse=True, initializer=self.default_initializer):
            outputs_fw, outputs_bw, _ = self.compute_rnn(_inputs, _lengths, True)

        with tf.variable_scope(self.name, reuse=reuse, initializer=self.default_initializer):
            if self.config.vocab_size <= 50000:
                softmax_fw = layers.Softmax(inputs=outputs_fw[:, 0: -1],
                                            labels=_inputs[:, 1:],
                                            vocab_size=self.config.vocab_size,
                                            name='fw/Softmax')
                softmax_bw = layers.Softmax(inputs=outputs_bw[:, 1:],
                                            labels=_inputs[:, 0: -1],
                                            vocab_size=self.config.vocab_size,
                                            name='bw/Softmax')
            else:
                # using the shortlist softmax
                softmax_fw = layers.ShortlistSoftmax(inputs=outputs_fw[:, 0: -1],
                                                     labels=_inputs[:, 1:],
                                                     shortlist=[10000, self.config.vocab_size],
                                                     name='fw/Softmax_Shortlist')
                softmax_bw = layers.ShortlistSoftmax(inputs=outputs_bw[:, 1:],
                                                     labels=_inputs[:, 0: -1],
                                                     shortlist=[10000, self.config.vocab_size],
                                                     name='bw/Softmax_Shortlist')

            # loss is the summation of all inputs
            max_len = tf.shape(_inputs)[1]
            loss_fw = softmax_fw.loss * tf.sequence_mask(_lengths-1, maxlen=max_len-1, dtype=tf.float32)
            loss_bw = softmax_bw.loss * tf.sequence_mask(_lengths-1, maxlen=max_len-1, dtype=tf.float32)
            loss = tf.reduce_sum(loss_fw + loss_bw) / 2

        # with tf.variable_scope(self.name, reuse=reuse, initializer=self.default_initializer):
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            # train = layers.TrainOp(loss, tvars, 'sgd', max_grad_norm=10, initial_lr=1.0, name='pretrain_op')
            train = layers.TrainOp(loss, tvars, 'adam', max_grad_norm=10, initial_lr=1e-3, name='pretrain_op')

            return train

    def get_copy_variables_to_propose_lstm(self, ref_vars):
        def filter_name(fullname):
            if fullname.find('bw') >= 0:
                return None
            if fullname.find('word_embedding') >= 0:
                return fullname[fullname.find('word_embedding'):]
            if fullname.find('multi_rnn_cell') >= 0:
                return fullname[fullname.find('multi_rnn_cell'):]
            if fullname.find('Softmax') >= 0:
                return fullname[fullname.find('Softmax'):]
            return None

        with tf.name_scope('copy'):
            copy_ops = []
            for v in self.pre_train.tvars:
                name1 = filter_name(v.name)
                for ref_v in ref_vars:
                    name2 = filter_name(ref_v.name)
                    if name1 == name2 and v.shape == ref_v.shape:
                        print('sync {} with {}'.format(v.name, ref_v.name))
                        copy_ops.append(tf.assign(ref_v, v))
                        break
        return copy_ops

    def run_pre_train(self, session, inputs, lengths, lr=1.0, is_update=True):

        if is_update:
            if lr is not None:
                self.pre_train.set_lr(session, lr)
            loss, _ = session.run([self.pre_train.loss, self.pre_train.train_op],
                                  {self._inputs: inputs, self._lengths: lengths})
        else:
            loss = session.run(self.pre_train.loss,
                               {self._inputs: inputs, self._lengths: lengths})

        return loss

    def run_pre_train_eval(self, session, seq_list, batch_size=10):

        sum_nlogp = 0
        sum_words = 0
        for seqs in reader.produce_batch_seqs(seq_list, batch_size):
            inputs, lengths = reader.produce_data_to_trf(seqs)

            sum_nlogp += self.run_pre_train(session, inputs, lengths, is_update=False)
            sum_words += np.sum(np.array(lengths) - 1)

        nll = sum_nlogp / len(seq_list)
        ppl = np.exp(sum_nlogp / sum_words)

        return nll, ppl

    def run_copy_vars(self, session):
        if self.copy_ops is not None:
            print('[{}] sync variables from trf to proposed_lstm'.format(self.name))
            session.run(self.copy_ops)

