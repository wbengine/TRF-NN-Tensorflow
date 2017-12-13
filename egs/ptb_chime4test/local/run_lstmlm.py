import tensorflow as tf
import time
import sys
import os
import numpy as np
import task

from base import *
from lm import *


def small_config(data):
    config = lstmlm.Config()
    config.vocab_size = data.get_vocab_size()
    config.embedding_size = 200
    config.hidden_size = 200
    config.hidden_layers = 2
    config.step_size = 20
    config.batch_size = 20
    config.epoch_num = 13
    config.init_weight = 0.1
    config.max_grad_norm = 5
    config.dropout = 0
    config.learning_rate = 1.0
    config.lr_decay = 0.5
    config.lr_decay_when = 4
    return config


def medium_config(data):
    config = lstmlm.Config()
    config.vocab_size = data.get_vocab_size()
    config.embedding_size = 650
    config.hidden_size = 650
    config.hidden_layers = 2
    config.step_size = 35
    config.batch_size = 100
    config.epoch_num = 39
    config.init_weight = 0.05
    config.max_grad_norm = 5
    config.dropout = 0.5
    config.learning_rate = 1.0
    config.lr_decay = 0.8
    config.lr_decay_when = 6
    return config


def large_config(data):
    config = lstmlm.Config()
    config.vocab_size = data.get_vocab_size()
    config.embedding_size = 1500
    config.hidden_size = 1500
    config.hidden_layers = 2
    config.step_size = 35
    config.batch_size = 20
    config.epoch_num = 55
    config.init_weight = 0.04
    config.max_grad_norm = 10
    config.dropout = 0.65
    config.learning_rate = 1.0
    config.lr_decay = 1 / 1.15
    config.lr_decay_when = 14
    return config


def create_name(config):
    s = str(config)
    return s


def main(_):
    data = reader.Data().load_raw_data([task.train, task.valid, task.test],
                                       add_beg_token=None,
                                       add_end_token='</s>',
                                       add_unknwon_token='<unk>')
    nbest_cmp = task.NBestComputer()
    res_file = 'results.txt'

    config = small_config(data)
    # config = medium_config(data)
    # config = large_config(data)

    work_dir = './lstm/' + create_name(config)
    wb.mkdir(work_dir, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(work_dir, 'lstm.log'))
    print(work_dir)
    wb.pprint_dict(config.__dict__)

    data.write_vocab(work_dir + '/vocab.txt')
    data.write_data(data.datas[1], work_dir + '/valid.id')
    data.write_data(data.datas[2], work_dir + '/test.id')

    write_model = os.path.join(work_dir, 'model.ckpt')

    with tf.Graph().as_default():
        # lm = lstmlm.FastLM(config, device_list=['/gpu:0', '/gpu:0'])
        lm = lstmlm.LM(config, device='/gpu:0')

        sv = tf.train.Supervisor(logdir=os.path.join(work_dir, 'logs'),
                                 summary_op=None, global_step=lm.global_step())
        sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:

            lm.train(session, data, write_model,
                     write_to_res=(res_file, create_name(config)),
                     is_shuffle=False)

            # rescore
            print('rescoring...')
            time_beg = time.time()
            for nbest in nbest_cmp.nbests:
                list = nbest.get_nbest_list(data)
                nbest.lmscore = lm.rescore(session, nbest.get_nbest_list(data))
                # print(len(nbest.lmscore))
            print('rescore time={:.2f}m'.format((time.time() - time_beg) / 60))
            nbest_cmp.write_lmscore(work_dir + '/model')

            # tune lm-scale
            print('computing wer...')
            nbest_cmp.cmp_wer()
            nbest_cmp.write_to_res(res_file, create_name(config))

if __name__ == '__main__':
    tf.app.run(main=main)
