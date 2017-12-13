import tensorflow as tf
import os
import sys
import time
import numpy as np

import task
from model import wblib as wb
from model import reader
from model import trfbase
from model import trfnce
from model import lstmlm
import run_lstmlm


# [data]
data = reader.Data().load_raw_data([task.train, task.valid, task.test], add_beg_token='<s>', add_end_token='</s>')


def create_name(config, q_config):
    s = str(config)
    if q_config is not None:
        s += '_with_' + run_lstmlm.create_name(q_config)
    return s


def main(_):
    config = trfnce.Config(data)

    config.structure_type = 'mix'
    config.embedding_dim = 128
    config.cnn_filters = [(i, 128) for i in range(1, 5)]
    config.cnn_hidden = 128
    config.cnn_layers = 1
    config.cnn_skip_connection = False
    config.cnn_residual = True
    config.cnn_activation = 'relu'
    config.rnn_hidden_layers = 1
    config.rnn_hidden_size = 128
    config.attention = True

    config.batch_size = 100
    config.noise_factor = 2
    config.noise_sampler = 2
    config.init_weight = 0.1
    config.optimize_method = ['sgd', 'sgd']
    config.lr_param = trfbase.LearningRateEpochDelay(1e-2, 0.5)
    config.lr_zeta = trfbase.LearningRateEpochDelay(1e-2, 0.5)
    config.max_epoch = 10
    # config.dropout = 0.75
    # config.init_zeta = config.get_initial_logz(0)
    config.update_zeta = True
    config.write_dbg = False
    config.pprint()

    q_config = run_lstmlm.small_config(data)
    # q_config = None

    name = create_name(config, q_config)
    logdir = 'trf_nce/' + name
    wb.mkdir(logdir, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(logdir, 'trf.log'))
    print(logdir)

    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    # wb.rmdir(logdirs)
    with tf.Graph().as_default():
        if q_config is None:
            m = trfnce.TRF(config, data, logdir=logdir, device='/gpu:0')
        else:
            m = trfnce.TRF(config, data, logdir=logdir, device='/gpu:1',
                           q_model=lstmlm.LM(q_config, device='/gpu:1')
                           )
        # noise_lstm = lstmlm.LM(run_lstmlm_withBegToken.small_config(data), device='/gpu:1')
        # m.lstm = noise_lstm

        sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                                 global_step=m.train_net.global_step)
        sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:
            m.set_session(session)

            if m.q_model is not None:
                print('load lstmlm for q model')
                m.q_model.restore(session, './lstm/' + run_lstmlm.create_name(q_config) + '/model.ckpt')

            m.train(sv, session,
                    print_per_epoch=0.1,
                    operation=task.Ops(m),
                    # nbest=nbest,
                    # lmscale_vec=np.linspace(1, 20, 20)
                    )

if __name__ == '__main__':
    tf.app.run(main=main)
