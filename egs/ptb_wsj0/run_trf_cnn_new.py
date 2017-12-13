import tensorflow as tf
import sys
import os
import numpy as np
import time

from model import reader
from model import trfnnbase as trf
from model import wblib as wb

# [data]
data = reader.Data().load_raw_data(reader.ptb_raw_dir(), add_beg_token='<s>', add_end_token='</s>')
# data.cut_train_to_length(50)
nbest = reader.NBest(*reader.wsj0_nbest())
nbest_list = data.load_data(nbest.nbest, is_nbest=True)


def create_name(config):
    return 'trf_' + str(config.config_trf) + '_maxlen{}'.format(config.max_len)


def get_config():
    config = trf.Config(data, 'cnn')
    config.jump_width = 2
    config.chain_num = 20
    config.batch_size = 100
    config.lr_cnn = trf.trfbase.LearningRateTime(beta=1., tc=1e4)
    config.lr_zeta = trf.trfbase.LearningRateTime(1.0, 0.2)

    config_trf = config.config_trf
    config_trf.opt_method = 'adam'
    config_trf.embedding_dim = 200
    config_trf.cnn_banks = [(i, 200) for i in range(1, 11)]
    config_trf.cnn_stacks = [(3, 200), (3, 400), (3, 800)]
    config_trf.cnn_skip_connection = True
    config_trf.cnn_activation = 'relu'
    config_trf.max_epoch = 100
    config_trf.init_weight = 0.1
    config_trf.max_grad_norm = 10
    config_trf.zeta_gap = 10
    config_trf.train_batch_size = 1000
    config_trf.sample_batch_size = 200
    config_trf.dropout = 0

    config_lstm = config.config_lstm
    config_lstm.hidden_size = 200
    config_lstm.step_size = 100
    # config_lstm.softmax_type = 'Shortlist'
    # config_lstm.adaptive_softmax_cutoff = [5000, config.vocab_size]

    return config


def main(_):

    # print(data.datas[0][0: 10])
    # print(data.get_max_len())
    # return

    config = get_config()
    name = create_name(config)
    logdir = wb.mkdir('./trf_cnn_new/' + name, is_recreate=True)
    sys.stdout = wb.std_log(logdir + '/trf.log')
    config.pprint()
    print(logdir)

    # word-embedding
    # config.config_trf.load_embedding_path = os.path.join(logdir, 'word_emb.txt')
    # config.config_trf.update_embedding = False
    # data.word2vec(config.config_trf.load_embedding_path, config.config_trf.embedding_dim, 0)

    # write data
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[0], logdir + '/train.id')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')
    data.write_data(nbest_list, logdir + '/nbest.id')

    with tf.Graph().as_default():
        m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')

        sv = tf.train.Supervisor(logdir=logdir + '/logs', summary_op=None, global_step=m.global_steps)
        # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:
            m.set_session(session)

            m.pre_train(sv, session, batch_size=100, max_epoch=3, lr=0.1)

            m.train(sv, session,
                    print_per_epoch=0.1,
                    nbest=nbest,
                    nbest_list=nbest_list)


if __name__ == '__main__':
    # pretrain()
    tf.app.run(main=main)
