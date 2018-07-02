import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from hrf import *


def main():
    with open('data.info') as f:
        data_info = json.load(f)

    data = seq.Data(vocab_files=data_info['vocab'],
                    train_list=data_info['train'],
                    valid_list=data_info['valid'],
                    test_list=data_info['test']
                    )

    config = crf.Config(data)

    # features
    config.mix_config.embedding_size = 200
    config.mix_config.hidden_size = 200
    config.mix_config.hidden_layers = 1
    config.mix_config.dropout = 0
    config.train_batch_size = 100

    config.lr_tag = lr.LearningRateEpochDelay(1e-2)
    config.lr_mix = lr.LearningRateEpochDelay(1e-2)
    config.opt_tag = 'momentum'
    config.opt_mix = 'momentum'
    config.max_epoch = 100
    config.print()

    logdir = wb.mklogdir('crf/' + str(config) + '_%s' % config.opt_mix, is_recreate=True)

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.tag'))
    data.write_file(data.datas[1], os.path.join(logdir, 'valid.id'))
    data.write_file(data.datas[2], os.path.join(logdir, 'test.id'))

    m = crf.CRF(config, data, logdir, device='/gpu:1')

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            ops = crf.DefaultOps(m, data.datas[1], data.datas[2])
            ops.perform_next_epoch = 0
            m.train(0.1, ops)


if __name__ == '__main__':
    main()
