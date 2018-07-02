import tensorflow as tf
import time
import sys
import json
import os
import numpy as np

from eval import nbest_eval_wod

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
    config.embedding_size = 512
    config.hidden_size = 512
    config.hidden_layers = 2
    config.step_size = 35
    config.batch_size = 20
    config.epoch_num = 10
    config.init_weight = 0.05
    config.max_grad_norm = 5
    config.dropout = 0.5
    config.learning_rate = 1.0
    config.lr_decay = 1.0
    config.lr_decay_when = 10
    return config


def large_config(data):
    config = lstmlm.Config()
    config.vocab_size = data.get_vocab_size()
    config.embedding_size = 1500
    config.hidden_size = 1500
    config.hidden_layers = 2
    config.step_size = 35
    config.batch_size = 20
    config.epoch_num = 20
    config.init_weight = 0.04
    config.max_grad_norm = 10
    config.dropout = 0.65
    config.learning_rate = 1.0
    config.lr_decay = 1
    config.lr_decay_when = 20
    return config


def create_name(config):
    s = str(config)
    return s


with open('data.info') as f:
    info = json.load(f)


def main(_):
    data = reader.Data().load_raw_data(file_list=info['hkust_train_wod'] +
                                                 info['hkust_valid_wod'] +
                                                 info['hkust_valid_wod'],
                                       add_beg_token=None,
                                       add_end_token='</s>',
                                       add_unknwon_token='<unk>')

    config = small_config(data)
    # config = medium_config(data)
    # config = large_config(data)

    work_dir = './lstm/' + create_name(config)
    wb.prepare_log_dir(work_dir, 'lstm.log')
    config.print()

    data.write_vocab(work_dir + '/vocab.txt')
    data.write_data(data.datas[0], work_dir + '/train.id')
    data.write_data(data.datas[1], work_dir + '/valid.id')
    data.write_data(data.datas[2], work_dir + '/test.id')

    write_model = os.path.join(work_dir, 'model.ckpt')

    # lm = lstmlm.FastLM(config, device_list=['/gpu:0', '/gpu:0'])
    lm = lstmlm.LM(config, data, device='/gpu:0')

    sv = tf.train.Supervisor(logdir=os.path.join(work_dir, 'logs'),
                             summary_op=None, global_step=lm.global_step())
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:

        lm.train(session, data,
                 write_model=write_model,
                 write_to_res=('results.txt', create_name(config)),
                 is_shuffle=False)

        print('compute the WER...')
        nbest_eval_wod(lm, data, work_dir, 'results.txt', create_name(config),
                       rescore_fun=lambda x: lm.rescore(session, x))

if __name__ == '__main__':
    tf.app.run(main=main)
