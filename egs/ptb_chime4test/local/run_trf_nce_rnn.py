import tensorflow as tf
import os
import sys
import time
import numpy as np

import task
from base import *
from lm import *
from trf import *
import run_lstmlm


# [data]
data = reader.Data().load_raw_data([task.train, task.valid, task.test],
                                   add_beg_token='</s>', add_end_token='</s>')
# data.cut_train_to_length(50)


def create_name(config):
    s = str(config)
    s += '_sgd'
    return s


def main(_):
    config = trfnce.Config(data)
    config.structure_type = 'rnn'
    config.embedding_dim = 200
    config.rnn_hidden_layers = 2
    config.rnn_hidden_size = 200
    config.rnn_type = 'lstm'
    config.rnn_predict = True
    config.batch_size = 10
    config.noise_factor = 10
    config.noise_sampler = 'lstm:lstm/lstm_e200_h200x2/model.ckpt'
    config.init_weight = 0.1
    config.optimize_method = ['sgd', 'sgd']
    config.lr_param = trfbase.LearningRateEpochDelay(1e-1)
    config.lr_zeta = trfbase.LearningRateEpochDelay(1e-1)
    config.max_epoch = 100
    # config.dropout = 0.75
    # config.init_zeta = config.get_initial_logz(0)
    config.update_zeta = True
    config.write_dbg = False
    config.print()

    name = create_name(config)
    logdir = 'trf_nce/' + name
    wb.mkdir(logdir, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(logdir, 'trf.log'))
    print(logdir)

    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    # wb.rmdir(logdirs)
    with tf.Graph().as_default():

        m = trfnce.TRF(config, data, logdir=logdir, device='/gpu:0')

        # s1 = trfnce.NoiseSamplerNgram(config, data, 2)
        # s2 = trfnce.NoiseSamplerLSTMEval(config, data, config.noise_sampler.split(':')[-1])

        sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                                 global_step=m.train_net.global_step)
        sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:
            with session.as_default():

                m.initialize(session)

                m.train(sv, session,
                        print_per_epoch=0.1,
                        operation=task.Ops(m),
                        )

            # with session.as_default():
            #     s1.start()
            #     s2.start()
            #
            #
            #     print(np.mean(s2.lstm.rescore(session, data.datas[1], pad_end_token_to_head=False)))
            #
            #     for i, s in enumerate([s1, s2]):
            #         nll_train = -np.mean(s.noise_logps(data.datas[0]))
            #         nll_valid = -np.mean(s.noise_logps(data.datas[1]))
            #         print('i={} nll_train={:.2f} nll_valid={:.2f}'.format(i, nll_train, nll_valid))
            #
            #     s1.release()
            #     s2.release()

if __name__ == '__main__':
    tf.app.run(main=main)
