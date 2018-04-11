import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from lm import *
from trf.common import net
from trf.sa import trf

from run_trf_neural_nce import Operation


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


def create_config(data):
    config = trf.Config(data)

    config.chain_num = 10
    config.multiple_trial = 10
    config.jump_width = 3
    config.sample_sub = 3
    config.train_batch_size = 1000
    config.sample_batch_size = 100
    config.auxiliary_type = 'lstm'
    config.auxiliary_config.embedding_size = 200
    config.auxiliary_config.hidden_size = 200
    config.auxiliary_config.hidden_layers = 1
    config.auxiliary_config.batch_size = 20
    config.auxiliary_config.step_size = 20
    config.auxiliary_config.learning_rate = 0.1

    config.lr_feat = lr.LearningRateEpochDelay(1e-3)
    config.lr_net = lr.LearningRateTime(1, 1, tc=1e4)
    config.lr_logz = lr.LearningRateTime(1.0, 0.2, tc=100)
    config.opt_feat_method = 'adam'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'adam'
    config.max_epoch = 1000

    # feat config
    # config.feat_config.feat_type_file = '../../tfcode/feat/g4.fs'
    config.feat_config = None

    # neural config
    config.net_config.update(get_config_cnn(config.vocab_size))

    # norm
    config.norm_type = 'linear'

    return config


def create_name(config):
    return str(config)


def main(_):
    data = reader.Data().load_raw_data(reader.ptb_raw_dir(),
                                       add_beg_token='<s>', add_end_token='</s>',
                                       add_unknwon_token='<unk>')

    # create config
    config = create_config(data)
    # create log dir
    logdir = 'trf_sa/' + create_name(config)
    # prepare the log dir
    wb.prepare_log_dir(logdir, 'trf.log')

    config.net_config.load_embedding_path = os.path.join(logdir, 'word_emb.txt')
    if config.net_config.load_embedding_path is not None:
        # get embedding vectors
        data.word2vec(config.net_config.load_embedding_path, config.net_config.embedding_dim, cnum=0)

    config.print()
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')
    # nce_pretrain_model_path = 'trf_nce/trf_nce20_e256_cnn_(1to10)x128_(3x128)x3_relu_noise2gram/trf.mod'

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                             global_step=m.global_step)
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            # m.restore_nce_model(nce_pretrain_model_path)
            m.train(operation=Operation(m))

if __name__ == '__main__':
    tf.app.run(main=main)
