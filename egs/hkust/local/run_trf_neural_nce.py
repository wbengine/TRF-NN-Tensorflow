import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from trf.common import net
from trf.isample import trf_nce as trf
# from trf.nce import trf

import task


class Operation(wb.Operation):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.perform_next_epoch = 0

    def perform(self, step, epoch):
        wer_real = task.nbest_eval(self.m, self.m.data, self.m.logdir,
                                   wb.FRes(os.path.join(self.m.logdir, 'wer_per_epoch.log')),
                                   'epoch%.2f' % epoch,
                                   )
        print('epoch={:.2f} wer_real={:.2f}'.format(
            epoch, wer_real))


def create_config(data):
    config = trf.Config(data)
    config.norm_config = 'linear'
    config.batch_size = 100
    config.noise_factor = 1
    config.noise_sampler = None  # '2gram'
    config.data_factor = 1
    config.data_sampler = config.noise_sampler

    # config.lr_feat = lr.LearningRateTime(1e-4)
    config.lr_net = lr.LearningRateTime(1e-3)  #lr.LearningRateTime(1, 0.5, tc=1e3)
    config.lr_logz = lr.LearningRateTime(0.01)
    config.opt_feat_method = 'adam'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'adam'
    config.max_epoch = 1000

    config.init_logz = config.get_initial_logz()
    config.init_global_logz = 0

    # config.prior_model_path = 'lstm/lstm_e32_h32x1_BNCE_SGD/model.ckpt'
    # feat config
    # config.feat_config.feat_type_file = '../../tfcode/feat/g4.fs'
    # config.feat_config.feat_cluster = None
    config.feat_config = None

    # net config
    config.net_config.update(task.get_config_rnn(config.vocab_size))
    # config.net_config.l2_reg = 1e-4
    # wb.mkdir('word_emb')
    # config.net_config.load_embedding_path = 'word_emb/ptb_d{}.emb'.format(config.net_config.embedding_dim)

    config.write_dbg = False

    return config


def create_name(config):
    return str(config)


def main(_):

    data = reader.Data().load_raw_data([task.train, task.valid, task.test],
                                       add_beg_token='</s>', add_end_token='</s>',
                                       add_unknwon_token='<unk>')

    # create config
    config = create_config(data)
    # create log dir
    logdir = 'trf_nce/' + create_name(config)
    # prepare the log dir
    wb.prepare_log_dir(logdir, 'trf.log')

    config.print()
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    if config.net_config.load_embedding_path is not None:
        # get embedding vectors
        data.word2vec(config.net_config.load_embedding_path, config.net_config.embedding_dim, cnum=0)

    # create TRF
    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                             global_step=m.global_step)
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            # train model
            m.train(operation=Operation(m))


if __name__ == '__main__':
    tf.app.run(main=main)
