import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from lm import *
from trf.isample import trf
from run_trf_neural_nce import get_config_cnn, get_config_rnn, Operation


def create_config(data):
    config = trf.Config(data)
    config.global_norm = True
    config.train_batch_size = 100
    config.sample_batch_size = 100

    config.sampler_config.learning_rate = 0.1

    config.lr_net = lr.LearningRateTime(1, 1, tc=1e5)
    config.lr_logz = lr.LearningRateTime(1, 0.2, tc=100)
    config.opt_feat_method = 'adam'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'sgd'
    config.max_epoch = 1000

    # feat config
    config.feat_config = None

    config.net_config.update(get_config_cnn(config.vocab_size))

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
    logdir = 'trf_is/' + create_name(config)
    # prepare the log dir
    wb.prepare_log_dir(logdir, 'trf.log')

    config.print()
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    config.net_config.load_embedding_path = os.path.join(logdir, 'word_emb.txt')
    if config.net_config.load_embedding_path is not None:
        # get embedding vectors
        data.word2vec(config.net_config.load_embedding_path, config.net_config.embedding_dim, cnum=0)

    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')
    # nce_pretrain_model_path = 'trf_nce/trf_nce20_e256_cnn_(1to10)x128_(3x128)x3_relu_noise2gram/trf.mod'

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                             global_step=m.global_step)
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.train(operation=Operation(m))


if __name__ == '__main__':
    tf.app.run(main=main)
