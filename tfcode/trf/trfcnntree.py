import tensorflow as tf
from copy import deepcopy
import numpy as np
import time
import os
import sys

from . import reader
from . import wblib as wb
from . import trfbase
from . import trf
from . import word2vec
from . import layers


class Net(trf.Net):
    def __init__(self, config, is_training, is_empirical, device, name='trf_phi', reuse=None):
        super().__init__(config, is_training, is_empirical, device, name, reuse)

    def output(self, config, reuse=None):

        vocab_size = config.vocab_size
        embedding_dim = config.embedding_dim
        tree_order = 2

        if config.cnn_activation is None:
            cnn_activation = tf.identity
        elif config.cnn_activation == 'relu':
            cnn_activation = tf.nn.relu
        elif config.cnn_activation == 'tanh':
            cnn_activation = tf.nn.tanh
        else:
            raise TypeError('unknown activation {}'.format(config.cnn_activation))

        batch_size = tf.shape(self._inputs)[0]
        max_len = tf.shape(self._inputs)[1]

        # embedding layers
        if config.load_embedding_path is not None:
            print('read init embedding vector from', config.load_embedding_path)
            emb_init_value = word2vec.read_vec(config.load_embedding_path)
            if emb_init_value.shape != (vocab_size, embedding_dim):
                raise TypeError('the reading embedding with shape ' +
                                str(emb_init_value.shape) +
                                ' does not match current shape ' +
                                str([vocab_size, embedding_dim]) +
                                '\nform path ' + config.load_embedding_path)
            word_embedding = tf.get_variable('word_embedding',
                                             [vocab_size, embedding_dim], dtype=tf.float32,
                                             initializer=tf.constant_initializer(emb_init_value),
                                             trainable=True)
        else:
            word_embedding = tf.get_variable('word_embedding',
                                             [vocab_size, embedding_dim], dtype=tf.float32)
        emb_output = tf.nn.embedding_lookup(word_embedding, self._inputs)  # (batch_size, seq_len, emb_dim)

        # pre-net
        pre_net_w = tf.get_variable('pre_net_w',
                                    [embedding_dim, config.cnn_hidden], dtype=tf.float32)
        pre_net_b = tf.get_variable('pre_net_b',
                                    [config.cnn_hidden], dtype=tf.float32)
        emb_output = tf.nn.relu(
            tf.matmul(tf.reshape(emb_output, [-1, embedding_dim]), pre_net_w) + pre_net_b)
        emb_output = tf.reshape(emb_output, [batch_size, max_len, config.cnn_hidden])
        emb_output = layers.embedding_mask(emb_output, self._lengths)

        tf.summary.image('embedding', tf.expand_dims(emb_output, axis=-1), max_outputs=4, collections=['cnn'])
        tf.summary.histogram('embedding', word_embedding, collections=['cnn'])

        ############################################
        # tree parser
        ############################################
        # variables
        tree_weight = tf.get_variable('tree_weight',
                                      [config.cnn_hidden], dtype=tf.float32)
        tree_filter = tf.get_variable('tree_filter',
                                      [tree_order, config.cnn_hidden, config.cnn_hidden], dtype=tf.float32)

        # condition of loop
        def cond_batch(e, n, phi, tree):
            # do parser if some sentence have not get the root node
            return tf.reduce_any(tf.greater(n, tree_order-1))

        # parser of loop
        def body_batch(e, n, phi, tree):

            y = tf.nn.conv1d(
                value=e,
                filters=tree_filter,
                stride=1,
                padding='VALID'
            )
            y = cnn_activation(y)

            new_n = n - (tree_order-1)
            old_max_len = tf.shape(e)[1]
            new_max_len = tf.shape(y)[1]

            old_len_range = tf.tile(tf.reshape(tf.range(old_max_len, dtype=tf.int32), [1, old_max_len]),
                                    [batch_size, 1])
            new_len_range = old_len_range[:, 0: new_max_len]
            len_mask = tf.less(new_len_range, tf.reshape(new_n, [batch_size, 1]))

            # calculate the weight of each sub-tree
            g = tf.matmul(tf.reshape(y, [-1, config.cnn_hidden]), tf.reshape(tree_weight, [-1, 1]))
            g = tf.reshape(g, [batch_size, -1])  # [batch_size, max_len]
            gmin = tf.reduce_min(g) - 1
            g = tf.where(len_mask, g, gmin * tf.ones_like(g))

            # get the the optimal sub-tree
            pos = tf.cast(tf.argmax(g, axis=-1), dtype=tf.int32)  # [batch_size]

            def get_mask(op, values, base):
                mask = op(values, tf.reshape(base, [batch_size, 1]))
                mask = tf.cast(mask, tf.float32)
                mask = tf.expand_dims(mask, axis=-1)
                return mask

            prefix_mask = get_mask(tf.less, old_len_range, pos)
            new_e_prefix = e * prefix_mask
            new_e_prefix = new_e_prefix[:, 0: -(tree_order-1)]  # to new_length

            pos_mask = get_mask(tf.equal, new_len_range, pos)
            new_e_pos = y * pos_mask  # new_length

            tail_mask = get_mask(tf.greater_equal, old_len_range, pos + 2)
            new_e_tail = e * tail_mask
            new_e_tail = new_e_tail[:, (tree_order-1):]  # to new_length

            new_e = new_e_prefix + new_e_pos + new_e_tail

            legial_bool = tf.cast(tf.greater(new_n, 0), dtype=tf.float32)
            new_phi = phi + layers.batch_pick(g, pos, batch_size) * legial_bool
            new_tree = tf.concat([tree, tf.reshape(pos, [-1, 1])], axis=1)

            return new_e, new_n, new_phi, new_tree

        init_phi = tf.zeros([batch_size], dtype=tf.float32, name='init_phi')
        init_tree = tf.reshape(tf.constant([], dtype=tf.int32, name='init_tree'), [batch_size, -1])

        final_e, final_n, final_phi, self.final_tree = \
            tf.while_loop(cond=cond_batch,
                          body=body_batch,
                          loop_vars=[
                              emb_output,
                              self._lengths,
                              init_phi,
                              init_tree],
                          shape_invariants=[
                              tf.TensorShape([None, None, config.cnn_hidden]),
                              # [batch_size, max_len, dim]
                              tf.TensorShape([None]),  # [batch_size]
                              tf.TensorShape([None]),  # [batch_size]
                              tf.TensorShape([None, None])  # [batch_size, tree_deep]
                          ])

        return final_phi

    def run_parser(self, session, inputs, lengths):
        nodes = session.run(self.final_tree,
                            {self._inputs: inputs,
                             self._lengths: lengths})
        return reader.extract_data_from_trf(nodes, lengths-1)

# class Net(object):
#     def __init__(self, config, is_training, device, name='trf_phi', reuse=None):
#         self.is_training = is_training
#         self.config = config
#
#         vocab_size = config.vocab_size
#         embedding_dim = config.embedding_dim
#         tree_order = 2
#
#         initializer = tf.random_uniform_initializer(-config.init_weight, config.init_weight)
#         with tf.device(device), tf.variable_scope(name, reuse=reuse, initializer=initializer):
#             # inputs: of shape (batch_size, seq_len)
#             # lengths: of shape (batch_size,)
#             # extra_weight: of shape (batch_size,), used to input the weight of such as ngram features
#             self._inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
#             self._lengths = tf.placeholder(tf.int32, [None], name='lengths')
#             self._extra_weight = tf.placeholder(tf.float32, [None], name='extra_weight')
#
#             if config.cnn_activation is None:
#                 cnn_activation = tf.identity
#             elif config.cnn_activation == 'relu':
#                 cnn_activation = tf.nn.relu
#             elif config.cnn_activation == 'tanh':
#                 cnn_activation = tf.nn.tanh
#             else:
#                 raise TypeError('unknown activation {}'.format(config.cnn_activation))
#
#             batch_size = tf.shape(self._inputs)[0]
#             max_len = tf.shape(self._inputs)[1]
#
#             # embedding layers
#             if config.load_embedding_path is not None:
#                 print('read init embedding vector from', config.load_embedding_path)
#                 emb_init_value = word2vec.read_vec(config.load_embedding_path)
#                 if emb_init_value.shape != (vocab_size, embedding_dim):
#                     raise TypeError('the reading embedding with shape ' +
#                                     str(emb_init_value.shape) +
#                                     ' does not match current shape ' +
#                                     str([vocab_size, embedding_dim]) +
#                                     '\nform path ' + config.load_embedding_path)
#                 word_embedding = tf.get_variable('word_embedding',
#                                                  [vocab_size, embedding_dim], dtype=tf.float32,
#                                                  initializer=tf.constant_initializer(emb_init_value),
#                                                  trainable=True)
#             else:
#                 word_embedding = tf.get_variable('word_embedding',
#                                                  [vocab_size, embedding_dim], dtype=tf.float32)
#             emb_output = tf.nn.embedding_lookup(word_embedding, self._inputs)  # (batch_size, seq_len, emb_dim)
#
#             # pre-net
#             pre_net_w = tf.get_variable('pre_net_w',
#                                         [embedding_dim, config.cnn_hidden], dtype=tf.float32)
#             pre_net_b = tf.get_variable('pre_net_b',
#                                         [config.cnn_hidden], dtype=tf.float32)
#             emb_output = tf.nn.relu(
#                 tf.matmul(tf.reshape(emb_output, [-1, embedding_dim]), pre_net_w) + pre_net_b)
#             emb_output = tf.reshape(emb_output, [batch_size, max_len, config.cnn_hidden])
#             emb_output = layers.embedding_mask(emb_output, self._lengths)
#
#             tf.summary.image('embedding', tf.expand_dims(emb_output, axis=-1), max_outputs=4, collections=['cnn'])
#             tf.summary.histogram('embedding', word_embedding, collections=['cnn'])
#
#             ############################################
#             # tree parser
#             ############################################
#             # variables
#             tree_weight = tf.get_variable('tree_weight',
#                                           [config.cnn_hidden], dtype=tf.float32)
#             tree_filter = tf.get_variable('tree_filter',
#                                           [tree_order, config.cnn_hidden, config.cnn_hidden], dtype=tf.float32)
#
#             # condition of loop
#             def cond_batch(e, n, phi, tree):
#                 # do parser if some sentence have not get the root node
#                 return tf.reduce_any(tf.greater(n, tree_order-1))
#
#             # parser of loop
#             def body_batch(e, n, phi, tree):
#
#                 y = tf.nn.conv1d(
#                     value=e,
#                     filters=tree_filter,
#                     stride=1,
#                     padding='VALID'
#                 )
#                 y = cnn_activation(y)
#
#                 new_n = n - (tree_order-1)
#                 old_max_len = tf.shape(e)[1]
#                 new_max_len = tf.shape(y)[1]
#
#                 old_len_range = tf.tile(tf.reshape(tf.range(old_max_len, dtype=tf.int32), [1, old_max_len]),
#                                         [batch_size, 1])
#                 new_len_range = old_len_range[:, 0: new_max_len]
#                 len_mask = tf.less(new_len_range, tf.reshape(new_n, [batch_size, 1]))
#
#                 # calculate the weight of each sub-tree
#                 g = tf.matmul(tf.reshape(y, [-1, config.cnn_hidden]), tf.reshape(tree_weight, [-1, 1]))
#                 g = tf.reshape(g, [batch_size, -1])  # [batch_size, max_len]
#                 gmin = tf.reduce_min(g) - 1
#                 g = tf.where(len_mask, g, gmin * tf.ones_like(g))
#
#                 # get the the optimal sub-tree
#                 pos = tf.cast(tf.argmax(g, axis=-1), dtype=tf.int32)  # [batch_size]
#
#                 def get_mask(op, values, base):
#                     mask = op(values, tf.reshape(base, [batch_size, 1]))
#                     mask = tf.cast(mask, tf.float32)
#                     mask = tf.expand_dims(mask, axis=-1)
#                     return mask
#
#                 prefix_mask = get_mask(tf.less, old_len_range, pos)
#                 new_e_prefix = e * prefix_mask
#                 new_e_prefix = new_e_prefix[:, 0: -(tree_order-1)]  # to new_length
#
#                 pos_mask = get_mask(tf.equal, new_len_range, pos)
#                 new_e_pos = y * pos_mask  # new_length
#
#                 tail_mask = get_mask(tf.greater_equal, old_len_range, pos + 2)
#                 new_e_tail = e * tail_mask
#                 new_e_tail = new_e_tail[:, (tree_order-1):]  # to new_length
#
#                 new_e = new_e_prefix + new_e_pos + new_e_tail
#
#                 legial_bool = tf.cast(tf.greater(new_n, 0), dtype=tf.float32)
#                 new_phi = phi + layers.batch_pick(g, pos, batch_size) * legial_bool
#                 new_tree = tf.concat([tree, tf.reshape(pos, [-1, 1])], axis=1)
#
#                 return new_e, new_n, new_phi, new_tree
#
#             init_phi = tf.zeros([batch_size], dtype=tf.float32, name='init_phi')
#             init_tree = tf.reshape(tf.constant([], dtype=tf.int32, name='init_tree'), [batch_size, -1])
#
#             self.final_e, \
#             self.final_n, \
#             self.final_phi, \
#             self.final_tree = tf.while_loop(cond=cond_batch,
#                                             body=body_batch,
#                                             loop_vars=[
#                                                 emb_output,
#                                                 self._lengths,
#                                                 init_phi,
#                                                 init_tree],
#                                             shape_invariants=[
#                                                 tf.TensorShape([None, None, config.cnn_hidden]),
#                                                 # [batch_size, max_len, dim]
#                                                 tf.TensorShape([None]),  # [batch_size]
#                                                 tf.TensorShape([None]),  # [batch_size]
#                                                 tf.TensorShape([None, None])  # [batch_size, tree_deep]
#                                             ])
#
#             self._phi = self.final_phi + self._extra_weight
#
#             # compute the gradient
#             # scalars: of shape [batch_sizs]
#             self._scalars = tf.placeholder(tf.float32, [None], name='scalars')
#             tvars = tf.trainable_variables()
#             # self.tvars = tvars
#             cost = tf.reduce_sum(self.final_phi * self._scalars)
#             grads = tf.gradients(cost, tvars)
#
#             # if training, create varialbes to store the gradients for given batches
#             if is_training:
#                 self.expectation = []  # a list of 'Tensor'
#                 self.clean_expec = []  # a list of 'operation' to set all the expectation to Zero
#                 self._update_expec = []  # a list of 'operation' to update the expectation
#                 for var, g in zip(tvars, grads):
#                     v = tf.Variable(np.zeros(var.shape, dtype='float32'), trainable=False)
#                     self.expectation.append(v)
#                     self.clean_expec.append(tf.assign(v, np.zeros(var.shape)))
#                     self._update_expec.append(tf.assign_add(v, g))
#
#             # pi and zeta
#             valid_len = config.max_len - config.min_len + 1
#             self._pi_0 = tf.get_variable('pi', shape=[valid_len], dtype=tf.float32,
#                                          trainable=False,
#                                          initializer=tf.constant_initializer(config.pi_0[config.min_len:]))
#             self._zeta = tf.get_variable('zeta', shape=[valid_len], dtype=tf.float32,
#                                          trainable=False,
#                                          initializer=tf.constant_initializer(config.initial_zeta()[config.min_len:]))
#             self._logz_base = tf.get_variable('logz_base', shape=[], dtype=tf.float32, trainable=False)
#             norm_constant = tf.log(self._pi_0) - self._zeta - self._logz_base
#             self._logp = self._phi + tf.gather(norm_constant, self._lengths - config.min_len)
#             # setting
#             self._new_pi_or_zeta = tf.placeholder(tf.float32, shape=[config.max_len + 1], name='new_pi_or_zeta')
#             self._set_pi_0 = tf.assign(self._pi_0, self._new_pi_or_zeta[config.min_len:])
#             self._set_zeta = tf.assign(self._zeta, self._new_pi_or_zeta[config.min_len:])
#             self._new_float = tf.placeholder(tf.float32, shape=[], name='new_float')
#             self._set_logz_base = tf.assign(self._logz_base, self._new_float)
#
#             # update zeta
#             self._sample_pi = tf.placeholder(tf.float32, shape=[config.max_len + 1], name='sample_pi')
#             self._zeta_lr = tf.placeholder(tf.float32, shape=[], name='zeta_lr')
#             zeta_step = tf.minimum(self._zeta_lr *
#                                    self._sample_pi[config.min_len:] / self._pi_0,
#                                    config.zeta_gap)
#             self._update_zeta = tf.assign_add(self._zeta, zeta_step)
#             self._update_zeta = tf.assign_sub(self._update_zeta,
#                                               self._update_zeta[0] * tf.ones_like(self._update_zeta))
#
#             # summary
#             tf.summary.scalar('logz_base', self._logz_base, collections=['cnn'])
#             self.summary_image = tf.summary.merge_all('cnn')
#
#     # def load_embedding(self, session):
#     #     if self.config.load_embedding_path is not None:
#     #         print('load embedding from', self.config.load_embedding_path)
#     #         self.loader.restore(session, self.config.load_embedding_path)
#
#     def get_logz(self, session):
#         logz_base = session.run(self._logz_base)
#         zeta = np.append(np.zeros(self.config.min_len),
#                          session.run(self._zeta) + logz_base)
#         return zeta
#
#     def get_zeta(self, session):
#         return np.append(np.zeros(self.config.min_len), session.run(self._zeta))
#
#     def set_pi(self, session, pi):
#         return session.run(self._set_pi_0, {self._new_pi_or_zeta: pi})
#
#     def set_zeta(self, session, zeta):
#         return session.run(self._set_zeta, {self._new_pi_or_zeta: zeta})
#
#     def set_logz_base(self, session, logz_base):
#         return session.run(self._set_logz_base, {self._new_float: logz_base})
#
#     def run_phi(self, session, inputs, lengths, extra_weight=None):
#         if extra_weight is None:
#             extra_weight = np.zeros(len(inputs), dtype='float32')
#         return session.run(self._phi,
#                            {self._inputs: inputs,
#                             self._lengths: lengths,
#                             self._extra_weight: extra_weight})
#
#     def run_logps(self, session, inputs, lengths, extra_weight=None):
#         if extra_weight is None:
#             extra_weight = np.zeros(len(inputs), dtype='float32')
#         return session.run(self._logp,
#                            {self._inputs: inputs,
#                             self._lengths: lengths,
#                             self._extra_weight: extra_weight})
#
#     def run_parser(self, session, inputs, lengths):
#         nodes = session.run(self.final_tree,
#                             {self._inputs: inputs,
#                              self._lengths: lengths})
#         return reader.extract_data_from_trf(nodes, lengths-1)
#
#     def run_update_expec(self, session, inputs, lengths, scalars):
#         session.run(self._update_expec,
#                     {self._inputs: inputs,
#                      self._lengths: lengths,
#                      self._scalars: scalars})
#
#     def run_update_zeta(self, session, sample_pi, zeta_lr):
#         session.run(self._update_zeta,
#                     {self._sample_pi: sample_pi,
#                      self._zeta_lr: zeta_lr})
#
#     def run_summary(self, session, inputs, lengths):
#         return session.run(self.summary_image,
#                            {self._inputs: inputs,
#                             self._lengths: lengths})


class TRF(trf.TRF):
    def __init__(self, config, data,
                 name='TRF', logdir='trf',
                 device='/gpu:0',
                 simulater_device=None):
        super().__init__(config, data, name=name, logdir=logdir,
                         device=device, simulater_device=simulater_device, network=Net)

        # with tf.name_scope(name + '_Model'):
        #     self.model_net = Net(config, is_training=True, is_empirical=False, device=device, name='phi_net', reuse=None)
        # with tf.name_scope(name + '_Empirical'):
        #     self.empir_net = Net(config, is_training=True, is_empirical=True, device=device, name='phi_net', reuse=True)
        #
        # with tf.device(device), tf.name_scope(name):
        #     # update paramters
        #     grads = []
        #     for (es, et) in zip(self.model_net.expectation, self.empir_net.expectation):
        #         grads.append(es - et)
        #
        #     self._lr_param = tf.Variable(0.001, trainable=False, name='learning_rate')  # learning rate
        #     self._global_step = tf.Variable(0, trainable=False, name='global_step')
        #
        #     if config.opt_method.lower() == 'adam':
        #         optimizer = tf.train.AdamOptimizer(self._lr_param)
        #     elif config.opt_method.lower() == 'var':
        #         beta = 0.9
        #         epsilon = config.var_gap
        #         # update empirical variance
        #         self.empirical_var = []
        #         self._update_empirical_var = []
        #         for exp, exp2 in zip(self.empir_net.expectation, self.empir_net.expectation2):
        #             v = tf.Variable(np.zeros(exp.shape, dtype='float32'), trainable=False)
        #             self.empirical_var.append(v)
        #             new_v = beta * v + (1 - beta) * (exp2 - exp * exp)
        #             self._update_empirical_var.append(tf.assign(v, new_v))
        #
        #         update_steps = [g / (var + epsilon) for g, var in zip(grads, self.empirical_var)]
        #         if config.max_grad_norm > 0:
        #             grads, grads_norm = tf.clip_by_global_norm(update_steps, config.max_grad_norm)
        #         optimizer = tf.train.GradientDescentOptimizer(self._lr_param)
        #
        #     else:
        #         if config.max_grad_norm > 0:
        #             grads, grads_norm = tf.clip_by_global_norm(grads, config.max_grad_norm)
        #         optimizer = tf.train.GradientDescentOptimizer(self._lr_param)
        #     # optimizer = tf.train.GradientDescentOptimizer(self._lr_param)
        #     tvars = tf.trainable_variables()
        #     self._train_op = optimizer.apply_gradients(zip(grads, tvars),
        #                                                global_step=self._global_step)
        #
        #     # update learining rate
        #     self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
        #     self._update_lr = tf.assign(self._lr_param, self._new_lr)
        #
        #     # summarys:
        #     self._param_size = tf.add_n([tf.size(v) for v in tf.trainable_variables()])
        #     self._param_norm = tf.global_norm(tf.trainable_variables())
        #     self._grad_norm = tf.global_norm(grads)
        #
        # # create simulater
        # self.create_simulater(simulater_device=device if simulater_device is None else simulater_device)

        # summarys
        self._pure_summary = tf.summary.merge([
            tf.summary.scalar('TRF/param_size', self._param_size),
            tf.summary.scalar('TRF/param_norm', self._param_norm),
            tf.summary.scalar('TRF/grad_norm', self._grad_norm),
        ]
        )

        # saver
        self.saver = tf.train.Saver()

        self.write_parser = open(os.path.join(logdir, name) + '.parser', 'wt')

    def train_after_update(self, **argv):
        eval_list = argv['eval_list']
        sv = argv['sv']
        session = argv['session']

        if eval_list is not None:
            summ = self.model_net.run_summary(session, *reader.produce_data_to_trf(eval_list))
            sv.summary_computed(session, summ)

        self.parser_output_tree(eval_list, f=self.write_parser)

    def output_tree(self, seq, parser_tree, f):
        assert len(seq) == len(parser_tree) + 1
        # create a tree
        node_list = []
        for n in seq:
            node_list.append(TreeNode('{}-{}'.format(self.data.word_list[n], n)))
        for i, n in enumerate(parser_tree):
            parent = TreeNode('parent-{}'.format(i), [node_list[n], node_list[n+1]])
            node_list[n] = parent
            del node_list[n+1]

        f.write('seq={}\nnod={}\n'.format(' '.join([self.data.word_list[n] for n in seq]), parser_tree))
        node_list[0].print(f)
        f.write('\n')
        f.flush()

    def parser_output_tree(self, seqs, f=None):
        if f is None:
            f = sys.stdout

        parser_trees = self.model_net.run_parser(self.get_session(), *reader.produce_data_to_trf(seqs))
        for seq, tree in zip(seqs, parser_trees):
            self.output_tree(seq, tree, f)


class TreeNode(object):
    def __init__(self, name, child_list=None):
        self.name = name
        self.child_list = child_list

    def print(self, f, level=0, IsTail=[]):
        if level == 0:
            s = ''
        else:
            s = ''
            for b in IsTail[0:-1]:
                s += '   ' if b else '│  '
            s += '└──' if IsTail[-1] else '├──'

        f.write(s + self.name + '\n')
        if self.child_list is not None:
            for child_node in self.child_list:
                child_node.print(f, level+1, IsTail+[child_node == self.child_list[-1]])


def main(_):

    # a = TreeNode('a')
    # b = TreeNode('b')
    # c = TreeNode('c', [a, b])
    # d1 = TreeNode('d1')
    # d2 = TreeNode('d2')
    # d = TreeNode('d', [d1, d2])
    # e = TreeNode('e', [c, d])
    # with open('../egs/word/test/parser.txt', 'wt') as f:
    #     e.print(f)
    # return

    data = reader.Data().load_raw_data(reader.word_raw_dir(),
                                       add_beg_token='<s>', add_end_token='</s>',
                                       add_unknwon_token=None)

    config = trfbase.Config(data)
    config.embedding_dim = 128
    # config.load_embedding_path = './embedding/emb_{}x{}.ckpt'.format(config.vocab_size, config.embedding_dim)
    config.pprint()

    # wb.rmdir(logdirs)
    with tf.Graph().as_default():
        m = TRF(config, data, logdir='../egs/word/test')

        sv = tf.train.Supervisor(logdir='../egs/word/test/logs', summary_op=None, global_step=m._global_step)
        sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs

        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:
            m.set_session(session)

            # m.test_sample()

            # train_seqs = data.datas[0][0: config.batch_size]
            # sample_seqs = train_seqs
            # _, summ = m.update(session, train_seqs, sample_seqs)
            #
            # sv.summary_computed(session, summ)

            # a = data.datas[0][0: 1]
            # print(a)
            # print(session.run(m.model_net._pi_0))
            # print(m.model_net.get_logz(session))
            # print(m.logps(*reader.produce_data_to_trf(a)))
            # print(m.phi(*reader.produce_data_to_trf(a)))

            s = ['organiation', 'application', 'applicances', 'banana']
            eval_list = data.load_data([[data.beg_token_str] + list(w) + [data.end_token_str] for w in s])
            print(eval_list)

            # trees = m.model_net.run_parser(session, *reader.produce_data_to_trf(eval_list))
            # print(trees)
            # m.parser_output_tree(eval_list)

            m.train(session, sv,
                    print_per_epoch=0.1,
                    eval_list=eval_list)


if __name__ == '__main__':
    tf.app.run(main=main)
