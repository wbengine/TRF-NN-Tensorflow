import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from semi import crf


train_num = 100

def main():
    with open('data.info') as f:
        data_info = json.load(f)

    data = seq.Data(vocab_files=data_info['vocab'],
                    train_list=data_info['train%d' % train_num],
                    valid_list=data_info['valid'],
                    test_list=data_info['test']
                    )

    config = crf.Config(data)

    # features
    config.mix_config.c2w_type = 'rnn'
    config.mix_config.chr_embedding_size = 100
    config.mix_config.c2w_rnn_size = 100
    config.mix_config.opt_method = 'adam'
    config.mix_config.dropout = 0.5

    config.train_batch_size = 20

    config.lr_mix = lr.LearningRateEpochDelay2(1e-3, delay=0.05)
    config.max_epoch = 100
    config.print()

    logdir = wb.mklogdir('train%d/crf/' % train_num + str(config) + '_%s' % config.mix_config.opt_method, is_recreate=True)

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.tag'))
    data.write_file(data.datas[1], os.path.join(logdir, 'valid.id'))
    data.write_file(data.datas[2], os.path.join(logdir, 'test.id'))

    m = crf.CRF(config, data, logdir, device='/gpu:0')

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            ops = crf.DefaultOps(m, data.datas[-2], data.datas[-1])
            m.train(0.1, ops)


if __name__ == '__main__':
    main()
