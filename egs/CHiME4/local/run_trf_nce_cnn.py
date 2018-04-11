import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from trf.nce import *
import task


def create_name(config):
    return str(config)+"_noise" + config.noise_sampler + '_onValidGen'


def main(_):
    data = reader.Data().load_raw_data([task.valid, task.valid, task.test],
                                       add_beg_token='</s>', add_end_token='</s>'
                                       )

    # seq = data.datas[0][0:3]
    # print(seq)
    # a, b = data.cut_data_to_length(seq, maxlen=10, minlen=6)
    # print(a)
    # print(b)
    # return

    config = trf.Config(data)
    config.max_epoch = 20
    config.batch_size = 20
    config.noise_factor = 10
    config.noise_sampler = '3gram'
    # config.init_logz = config.get_initial_logz(0)
    config.lr_feat = lr.LearningRateEpochDelay(1.0)
    config.lr_net = lr.LearningRateEpochDelay(1.0)
    config.lr_logz = lr.LearningRateEpochDelay(1e-3)
    config.opt_feat_method = 'sgd'
    config.opt_net_method = 'sgd'
    config.opt_logz_method = 'sgd'

    # config.prior_model_path = 'lstm/lstm_e32_h32x1_BNCE_SGD/model.ckpt'
    # feat config
    # config.feat_config.feat_type_file = '../../tfcode/feat/g3.fs'
    # config.feat_config.feat_cluster = None
    config.feat_config = None

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
    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')

    print('noise_nll=', -np.mean(m.noise_sampler.noise_logps(data.datas[0])))

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                             global_step=m.global_step)
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:

        with session.as_default():

            # print(m.eval(data.datas[1]))

            m.train(print_per_epoch=0.1,
                    operation=task.Ops(m))


if __name__ == '__main__':
    tf.app.run(main=main)
