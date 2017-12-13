import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from trf.nce import *
import task


def create_name(config):
    return str(config) + '_nopi'


def main(_):
    data = reader.Data().load_raw_data([task.train, task.valid, task.test],
                                       add_beg_token='</s>', add_end_token='</s>'
                                       )

    config = trf.Config(data)
    config.write_dbg = False
    config.max_epoch = 100
    config.batch_size = 10
    config.noise_factor = 10
    config.noise_sampler = '2gram'
    config.pi_true = np.ones_like(config.pi_true)

    config.init_logz = [np.log(config.vocab_size), np.log(config.vocab_size)]
    config.lr_feat = lr.LearningRateEpochDelay(1.0)
    config.lr_net = lr.LearningRateEpochDelay(1.0, delay=0.5, delay_when=4)
    config.lr_logz = lr.LearningRateEpochDelay(1e-3)  # using the empirical variance
    config.opt_feat_method = 'sgd'
    config.opt_net_method = 'sgd'
    config.opt_logz_method = 'adam'

    config.prior_model_path = 'lstm/lstm_e200_h200x2_BNCE_SGD/model.ckpt'
    # feat config
    config.feat_config = None
    # config.feat_config.feat_type_file = '../../../tfcode/feat/g3.fs'
    # config.feat_config.feat_cluster = None

    # net config
    # config.net_config = None
    # config.net_config.structure_type = 'mix'
    # config.net_config.embedding_dim = 128
    # config.net_config.cnn_filters = [(i, 128) for i in range(1, 11)]
    # config.net_config.cnn_hidden = 128
    # config.net_config.cnn_width = 3
    # config.net_config.cnn_layers = 3
    # config.net_config.cnn_skip_connection = False
    # config.net_config.cnn_residual = True
    # config.net_config.rnn_type = 'blstm'
    # config.net_config.rnn_hidden_layers = 1
    # config.net_config.rnn_hidden_size = 128
    # config.net_config.attention = True

    config.net_config.structure_type = 'cnn'
    config.net_config.embedding_dim = 128
    config.net_config.cnn_filters = [(i, 128) for i in range(1, 11)]
    config.net_config.cnn_hidden = 128
    config.net_config.cnn_width = 3
    config.net_config.cnn_layers = 3

    # config.net_config.structure_type = 'rnn'
    # config.net_config.rnn_type = 'brnn'
    # config.net_config.embedding_dim = 200
    # config.net_config.rnn_hidden_size = 200
    # config.net_config.rnn_hidden_layers = 2
    # config.net_config.rnn_predict = True

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

    # m.noise_sampler.start()
    # seqs, noise_logp = m.noise_sampler.get()
    # print(seqs)
    # print(noise_logp)
    # print('noise_nll=', -np.mean(m.noise_sampler.noise_logps(seqs)))
    # return

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                             global_step=m.global_step)
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:

        with session.as_default():

            # print(m.eval(data.datas[1]))

            m.train(print_per_epoch=0.1,
                    operation=task.Ops(m, 2))


if __name__ == '__main__':
    tf.app.run(main=main)
