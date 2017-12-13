import tensorflow as tf
import sys
import os
import numpy as np
import time

from base import *
from trf import trfrnn as trf
from trf import trfbase
import task
import run_lstmlm

# [data]
data = reader.Data().load_raw_data([task.train, task.valid, task.test],
                                    add_beg_token='</s>',
                                    add_end_token='</s>',
                                    add_unknwon_token='<unk>')


def create_name(config):
    s = str(config)
    return s


def get_config():

    config = trf.Config(data, 'lstm/lstm_e200_h200x2_BNCE_SGD/model.ckpt')

    config.max_epoch = 100
    config.init_weight = 0.1
    config.opt_method = 'adam'
    config.jump_width = 2
    config.chain_num = 10
    config.multiple_trial = 10
    config.train_batch_size = 100
    config.sample_batch_size = 10
    config.lr_cnn = trfbase.LearningRateEpochDelay(1e-4, 1/1.1, 10)
    # config.lr_param = trf.trfjsa.LearningRateTime(1, 1, tc=1e4)
    config.lr_zeta = trfbase.LearningRateEpochDelay(1.0, 1/1.1, 10)
    config.update_param = False

    config.auxiliary_model = 'lstm'
    config.auxiliary_config.embedding_size = 32
    config.auxiliary_config.hidden_size = 32
    config.auxiliary_config.hidden_layers = 1
    config.auxiliary_config.batch_size = 10
    config.auxiliary_config.step_size = 10
    config.auxiliary_config.learning_rate = 1.0

    # config.feat_type_file = '../../tfcode/feat/g3.fs'
    # config.feat_cluster = 200

    return config


def main(_):

    config = get_config()
    # q_config = run_lstmlm.small_config(data)
    # q_config = None
    name = create_name(config)
    logdir = wb.mkdir('./trf_cnn/' + name, is_recreate=True)
    sys.stdout = wb.std_log(logdir + '/trf.log')
    config.print()
    print(logdir)

    if config.feat_cluster is not None:
        data.word2vec(logdir + '/vocab.cluster', config.embedding_dim, config.feat_cluster)

    # write data
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[0], logdir + '/train.id')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    with tf.Graph().as_default():
        m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')

        sv = tf.train.Supervisor(logdir=logdir + '/logs', summary_op=None, global_step=m._global_step)
        # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:
            m.set_session(session)

            m.train(sv, session,
                    print_per_epoch=0.1,
                    operation=task.Ops(m),
                    model_per_epoch=None)


if __name__ == '__main__':
    # test_simulater()
    tf.app.run(main=main)
