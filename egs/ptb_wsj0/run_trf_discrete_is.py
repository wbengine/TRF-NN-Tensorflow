import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from lm import *
from trf.isample import trf, sampler
from trf.common import net


def create_config(data):
    config = trf.Config(data)
    config.sample_batch_size = 100

    config.sampler_config = sampler.Ngram.Config()

    config.lr_feat = lr.LearningRateTime(1, 1, tc=1e3)
    config.lr_net = lr.LearningRateTime(1, 1, tc=1e4)
    config.lr_logz = lr.LearningRateTime(1.0, 1.0, tc=1e2)
    config.opt_feat_method = 'adam'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'sgd'
    config.max_epoch = 100

    # feat config
    config.feat_config.feat_type_file = '../../tfcode/feat/g4.fs'
    config.feat_config.feat_cluster = 200
    config.feat_config.pre_compute_data_exp = True
    config.feat_config.L2_reg = 1e-5

    config.net_config = None

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

    if config.feat_config.feat_cluster is not None:
        # get embedding vectors
        data.word2vec(os.path.join(logdir, 'emb.txt'),
                      200,
                      cnum=config.feat_config.feat_cluster)

    config.print()
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')
    # nce_pretrain_model_path = 'trf_nce/trf_nce20_e256_cnn_(1to10)x128_(3x128)x3_relu_noise2gram/trf.mod'

    # sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
    #                          global_step=m.global_step)
    # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    # session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # session_config.gpu_options.allow_growth = True
    # with sv.managed_session(config=session_config) as session:
    #     with session.as_default():
    m.train(operation=trf.DefaultOps(m, reader.wsj0_nbest()))


if __name__ == '__main__':
    tf.app.run(main=main)
