import tensorflow as tf
import sys
import os
import numpy as np
import time

from base import *
from lm import *
from trf.sa import *
import task

# [data]
data = reader.Data().load_raw_data([task.train, task.valid, task.valid],
                                    add_beg_token='</s>',
                                    add_end_token='</s>',
                                    add_unknwon_token='<unk>')


def create_name(config):
    s = str(config)
    return s


def get_config():
    config = trf.Config(data)
    config.max_epoch = 1000
    config.chain_num = 10
    config.multiple_trial = 10
    config.train_batch_size = 1000
    config.sample_batch_size = 100

    config.auxiliary_config.embedding_size = 200
    config.auxiliary_config.hidden_layers = 1
    config.auxiliary_config.hidden_size = 200
    config.auxiliary_config.learning_rate = 1.0
    config.auxiliary_config.batch_size = 100

    config.feat_config.feat_type_file = '../../../tfcode/feat/g3.fs'
    # config.feat_cluster = 200

    config.lr_feat = lr.LearningRateEpochDelay(1e-3)
    config.lr_net = lr.LearningRateEpochDelay(1e-3)
    config.lr_logz = lr.LearningRateEpochDelay(0.1)
    config.opt_feat_method = 'adam'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'sgd'

    return config


def main(_):

    config = get_config()
    name = create_name(config)
    logdir = wb.mkdir('./trf_sa/' + name, is_recreate=True)
    sys.stdout = wb.std_log(logdir + '/trf.log')
    config.print()
    print(logdir)

    # write data
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[0], logdir + '/train.id')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')
    ops = task.Ops(m)

    sv = tf.train.Supervisor(logdir=logdir + '/logs', summary_op=None, global_step=m.global_step)
    # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.train(print_per_epoch=0.1,
                    operation=ops)


if __name__ == '__main__':
    tf.app.run(main=main)
