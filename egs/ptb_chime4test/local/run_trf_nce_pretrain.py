import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from trf.nce import *
from lm import *
import task


def create_name(config):
    return str(config) + '_pretrain'


def main(_):
    data = reader.Data().load_raw_data([task.train, task.valid, task.test],
                                       add_beg_token='</s>', add_end_token='</s>'
                                       )

    # pretrain_lstm_path = 'lstm/lstm_+withBegToken_200x2/model.ckpt'
    # lstm_config = lstmlm.load_config(pretrain_lstm_path)

    config = trf.Config(data)
    config.write_dbg = True
    config.max_epoch = 100
    config.batch_size = 10
    config.noise_factor = 10
    config.noise_sampler = '2gram_nopi'
    # config.pi_true = np.ones_like(config.pi_true)

    # config.lr_feat = lr.LearningRateEpochDelay(1.0)
    config.lr_net = lr.LearningRateEpochDelay(1e-3)
    config.lr_logz = lr.LearningRateEpochDelay(0)  # using the empirical variance
    # config.opt_feat_method = 'sgd'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'adam'
    config.init_logz = [0, 9]

    # feat config
    config.feat_config = None
    # config.feat_config.feat_type_file = '../../../tfcode/feat/g3.fs'
    # config.feat_config.feat_cluster = None

    # net config
    config.net_config.structure_type = 'rnn'
    config.net_config.rnn_type = 'lstm'
    config.net_config.embedding_dim = 200
    config.net_config.rnn_hidden_size = 200
    config.net_config.rnn_hidden_layers = 2
    config.net_config.rnn_predict = True

    name = create_name(config)
    logdir = 'trf_nce_new/' + name
    wb.mkdir(logdir, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(logdir, 'trf.log'))
    print(logdir)
    config.print()

    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    # wb.rmdir(logdirs)
    m = trf.TRF2(config, data, logdir=logdir, device='/gpu:0', pretrain_lstm_path=None)

    # lm = lstmlm.load(pretrain_lstm_path)

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                             global_step=m.global_step)
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:

        with session.as_default():

            # s = data.datas[0][100]
            #
            # m.initialize()
            # print(m.get_log_probs([s]))
            #
            # lm.restore(session)
            # print(-lm.rescore(session, [s], pad_end_token_to_head=False))


            # m.initialize()
            # print(m.get_log_probs(data.datas[1]))

            m.train(print_per_epoch=0.1,
                    operation=task.Ops(m, 1))


if __name__ == '__main__':
    tf.app.run(main=main)
