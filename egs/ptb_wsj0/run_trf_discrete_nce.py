import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from trf.common import net
from trf.nce import trf as trf
from run_trf_neural_sa import get_config_cnn, get_config_rnn


def create_config(data):
    config = trf.Config(data)
    config.max_epoch = 100
    config.batch_size = 100
    config.noise_factor = 10
    config.norm_config = 'linear'
    config.noise_sampler = '2gram'

    config.lr_feat = lr.LearningRateEpochDelay(1e-3)
    config.lr_logz = lr.LearningRateTime(1e-3)
    config.opt_feat_method = 'adam'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'adam'

    config.init_logz = [9, 9]

    # feat config
    config.feat_config.feat_type_file = '../../tfcode/feat/g4.fs'
    config.feat_config.feat_cluster = 200

    # net config
    config.net_config = None

    config.write_dbg = True

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
    logdir = 'trf_nce/' + create_name(config)
    # prepare the log dir
    wb.prepare_log_dir(logdir, 'trf.log')

    config.print()
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    if config.feat_config.feat_cluster is not None:
        # get embedding vectors
        data.word2vec(os.path.join(logdir, 'emb.txt'),
                      200,
                      cnum=config.feat_config.feat_cluster)

    # create TRF
    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')

    # sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
    #                          global_step=m.global_step)
    # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    # session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # session_config.gpu_options.allow_growth = True
    # with sv.managed_session(config=session_config) as session:
    #     with session.as_default():
            # train model

    m.train(operation=trf.DefaultOps(m, reader.wsj0_nbest()))


if __name__ == '__main__':
    tf.app.run(main=main)
