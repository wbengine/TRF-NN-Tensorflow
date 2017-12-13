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
data = reader.Data().load_raw_data([task.train, task.valid, task.test],
                                   add_beg_token='</s>', add_end_token='</s>')
# data.cut_train_to_length(50)


def create_name(config, q_config):
    s = str(config)
    if q_config is not None:
        s += '_with_' + run_lstmlm.create_name(q_config)

    # s += '_op%d' % config.noise_operation_num
    # s += '_lstm'
    # s += '_logz{}'.format(int(config.init_zeta[0]))
    return s


def main(_):
    config = trfnce.Config(data)
    config.structure_type = 'cnn'
    config.embedding_dim = 200
    config.cnn_filters = [(i, 100) for i in range(1, 11)]
    config.cnn_width = 3
    config.cnn_layers = 3
    config.cnn_hidden = 200
    config.rnn_hidden_layers = 2
    config.rnn_hidden_size = 200
    config.rnn_predict = True
    config.batch_size = 10
    config.noise_factor = 10
    config.noise_sampler = 'lstm:lstm/lstm_e200_h200x2/model.ckpt'
    config.init_weight = 0.1
    config.optimize_method = ['adam', 'adam']
    config.lr_param = trfbase.LearningRateEpochDelay(0.001)
    config.lr_zeta = trfbase.LearningRateEpochDelay(0.01)
    config.max_epoch = 100
    # config.dropout = 0.75
    # config.init_zeta = config.get_initial_logz(20)
    config.update_zeta = True
    config.write_dbg = False
    config.print()

    # q_config = run_lstmlm.small_config(data)
    q_config = None

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
            m = trfnce.TRF(config, data, logdir=logdir, device='/gpu:0',
                           q_model=lstmlm.LM(q_config, device='/gpu:0')
                           )

        # s1 = trfnce.NoiseSamplerNgram(config, data, 2)
        # s2 = trfnce.NoiseSamplerLSTMEval(config, data, config.noise_sampler.split(':')[-1])

        sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                                 global_step=m.train_net.global_step)
        sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:
            with session.as_default():

                if m.q_model is not None:
                    print('load lstmlm for q model')
                    m.q_model.restore(session, './lstm/' + run_lstmlm.create_name(q_config) + '/model.ckpt')

                m.train(sv, session,
                        print_per_epoch=0.1,
                        operation=task.Ops(m),
                        )

if __name__ == '__main__':
    tf.app.run(main=main)
