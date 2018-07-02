import tensorflow as tf
import numpy as np
import json
import os

from base import *
from semi import trf, nce_net


def main():
    with open('data.info') as f:
        data_info = json.load(f)

    train_num = -1
    data = seq.Data(vocab_files=data_info['vocab'],
                    train_list=data_info['google_1b'],
                    valid_list=data_info['valid'],
                    test_list=data_info['test'],
                    max_len=60
                    )

    data_full = seq.Data(vocab_files=data_info['vocab'],
                         train_list=data_info['train'] if train_num == -1 else data_info['train%d' % train_num],
                         valid_list=data_info['valid'],
                         test_list=data_info['test']
                         )

    config = trf.Config(data)
    config.mix_config.c2w_type = 'cnn'
    config.mix_config.chr_embedding_size = 30
    config.mix_config.c2w_cnn_size = 30
    config.mix_config.opt_method = 'adam'
    config.mix_config.dropout = 0.5
    config.crf_batch_size = 100
    config.trf_batch_size = 100
    config.data_factor = 0

    config.lr = lr.LearningRateEpochDelay(1e-3)

    logdir = 'trainall/' if train_num == -1 else 'train%d/' % train_num
    logdir = wb.mklogdir(logdir + str(config), is_recreate=True)
    config.print()

    m = trf.TRF(config, data, data_full, logdir, device='/gpu:1')

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            # print(m.get_crf_logps(data_full.datas[0][0: 10]))
            # print(m.get_trf_logps(data_full.datas[0][0: 10]))
            ops = trf.DefaultOps(m, data.datas[-2], data.datas[-1], data_info['nbest'])
            # ops.perform(0, 0)
            m.train(0.1, ops)


if __name__ == '__main__':
    main()