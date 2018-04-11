import tensorflow as tf
import os
import sys
import numpy as np
import time

from base import *
from trf.common import net
from trf.isample import trf_nce as trf
# from trf.nce import trf

import task
nbest_cmp = task.NBestComputer()


def get_config_cnn(vocab_size):
    config = net.Config(vocab_size)
    config.embedding_dim = 256
    config.structure_type = 'cnn'
    config.cnn_filters = [(i, 128) for i in range(1, 11)]
    config.cnn_hidden = 128
    config.cnn_width = 3
    config.cnn_layers = 3
    config.cnn_activation = 'relu'
    config.cnn_skip_connection = True
    config.cnn_residual = False
    config.cnn_batch_normalize = False
    config.cnn_final_activation = None
    return config


def get_config_rnn(vocab_size):
    config = net.Config(vocab_size)
    config.embedding_dim = 200
    config.structure_type = 'rnn'
    config.rnn_type = 'blstm'
    config.rnn_hidden_size = 200
    config.rnn_hidden_layers = 1
    config.rnn_predict = True
    config.rnn_share_emb = True
    return config


def get_config_mix(vocab_size):
    config = net.Config(vocab_size)
    config.embedding_dim = 200
    config.structure_type = 'mix'

    config.cnn_filters = [(i, 128) for i in range(1, 11)]
    config.cnn_hidden = 128
    config.cnn_width = 3
    config.cnn_layers = 3
    config.cnn_activation = 'relu'
    config.cnn_skip_connection = True
    config.cnn_residual = False
    config.cnn_batch_normalize = False

    config.rnn_type = 'blstm'
    config.rnn_hidden_size = 128
    config.rnn_hidden_layers = 1
    # config.rnn_predict = True
    # config.rnn_share_emb = True

    config.attention = True
    return config


def get_config_rnn_large(vocab_size):
    config = net.Config(vocab_size)
    config.embedding_dim = 512
    config.structure_type = 'rnn'
    config.rnn_type = 'blstm'
    config.rnn_hidden_size = 512
    config.rnn_hidden_layers = 2
    config.rnn_predict = True
    config.rnn_share_emb = True
    return config


class Operation(wb.Operation):
    def __init__(self, trf_model):
        super().__init__()
        self.perform_next_epoch = 0
        self.perform_per_epoch = 0.2

        self.m = trf_model
        self.opt_det_wer = 100
        self.opt_txt_wer = 100
        self.write_models = wb.mkdir(os.path.join(self.m.logdir, 'wer_models'))
        self.write_lmscore = wb.mkdir(os.path.join(self.m.logdir, 'lmscores'))

    def perform(self, step, epoch):
        super().perform(step, epoch)

        epoch_num = int(epoch / self.perform_per_epoch) * self.perform_per_epoch
        print('[Ops] rescoring:', end=' ', flush=True)  # resocring
        with self.m.time_recoder.recode('rescore'):
            time_beg = time.time()
            for nbest in nbest_cmp.nbests:
                nbest.lmscore = self.m.rescore(nbest.get_nbest_list(self.m.data))
            rescore_time = time.time() - time_beg
        nbest_cmp.write_lmscore(os.path.join(self.write_lmscore, 'epoch%.2f' % epoch_num))
        # compute wer
        with self.m.time_recoder.recode('wer'):
            time_beg = time.time()
            nbest_cmp.cmp_wer()
            nbest_cmp.write_to_res(os.path.join(self.m.logdir, 'wer_per_epoch.log'), 'epoch%.2f' % epoch_num)
            dev_wer = nbest_cmp.get_valid_wer()
            tst_wer = nbest_cmp.get_test_wer()
            wer_time = time.time() - time_beg
            print('epoch={:.2f} dev_wer={:.2f} test_wer={:.2f} lmscale={} '
                  'rescore_time={:.2f}, wer_time={:.2f}'.format(
                epoch, dev_wer, tst_wer, nbest_cmp.lmscale,
                rescore_time / 60, wer_time / 60))

        # write models
        if dev_wer < self.opt_det_wer:
            self.opt_det_wer = dev_wer
            self.m.save(self.write_models + '/epoch%.2f' % epoch_num)


def create_config(data):
    config = trf.Config(data)
    # config.pi_0 = data.get_pi0(config.pi_true)
    # config.pi_true = config.pi_0
    config.norm_config = 'linear'
    config.batch_size = 100
    config.noise_factor = 1
    config.noise_sampler = '2gram'
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
    config.net_config.update(get_config_rnn_large(config.vocab_size))
    # config.net_config.l2_reg = 1e-4
    # wb.mkdir('word_emb')
    # config.net_config.load_embedding_path = 'word_emb/ptb_d{}.emb'.format(config.net_config.embedding_dim)

    config.write_dbg = False

    return config


def create_name(config):
    return str(config)


def main(_):

    data = reader.Data().load_raw_data([task.train, task.valid, task.valid],
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
    m = trf.TRF(config, data, logdir=logdir, device='/gpu:1')

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                             global_step=m.global_step)
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            # train model
            m.train(print_per_epoch=0.01, operation=Operation(m))

            # seq_list = [[0, 10, 10, 1], [0, 20, 20, 20, 20, 20, 20, 20, 1]]
            # print(seq_list)
            # seq_list = m.sampler.lstm.add_noise(session, seq_list)
            # print(seq_list)



if __name__ == '__main__':
    tf.app.run(main=main)
