import tensorflow as tf
import sys
import os
import numpy as np
import time

from model import reader
from model import trfrnn as trf
from model import wblib as wb

# [data]
data = reader.Data().load_raw_data(reader.ptb_raw_dir(), add_beg_token='<s>', add_end_token='</s>')
nbest = reader.NBest(*reader.wsj0_nbest())
nbest_list = data.load_data(reader.wsj0_nbest()[0], is_nbest=True)


def create_name(config):
    s = 'trf_rnn'
    s += '_e{}'.format(config.embedding_dim)
    s += '_{}x{}'.format(config.hidden_size, config.hidden_layers)
    s += '_' + config.auxiliary_model
    return s


def get_config():
    config = trf.Config(data)
    # config.forward_lstm_path = 'trf_rnn/lstm{}x{}/model.ckpt'.format(
    #     config.hidden_size, config.hidden_layers)
    # config.backward_lstm_path = 'trf_rnn/inv_lstm{}x{}/model.ckpt'.format(
    #     config.hidden_size, config.hidden_layers)
    config.max_epoch = 1000
    config.init_weight = 0.1
    config.opt_method = 'adam'
    config.max_grad_norm = 10
    config.train_batch_size = 500
    config.sample_batch_size = 500
    config.jump_width = 1
    config.chain_num = 10
    config.sample_sub = 10
    config.multiple_trial = 10
    config.lr_cnn = trf.trfbase.LearningRateTime(1, 1, tc=1e4)
    config.lr_zeta = trf.trfbase.LearningRateTime(1.0, 0.2)
    config.auxiliary_lr = 0.1
    return config


def main(_):
    config = get_config()
    # config.auxiliary_shortlist = [4000, config.vocab_size]
    # config.sample_sub = 100
    # config.multiple_trial = 10
    name = create_name(config)
    logdir = wb.mkdir('./trf_rnn/' + name, is_recreate=True)
    sys.stdout = wb.std_log(logdir + '/trf.log')
    config.pprint()
    print(logdir)

    # write data
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[0], logdir + '/train.id')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')
    data.write_data(nbest_list, logdir + '/nbest.id')

    with tf.Graph().as_default():
        m = trf.TRF(config, data, logdir=logdir,
                    device='/gpu:0',
                    simulater_device='/gpu:0')

        sv = tf.train.Supervisor(logdir=logdir + '/logs', summary_op=None, global_step=m._global_step)
        # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:
            m.set_session(session)

            m.train(sv, session,
                    print_per_epoch=0.1,
                    nbest=nbest,
                    nbest_list=nbest_list)


if __name__ == '__main__':
    # pretrain()
    tf.app.run(main=main)
