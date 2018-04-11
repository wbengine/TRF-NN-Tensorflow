import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from hrf import *


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
    config.train_batch_size = 10
    config.lr_tag = lr.LearningRateEpochDelay(0.01)
    config.lr_mix = lr.LearningRateEpochDelay(0.01)
    config.opt_tag = 'momentum'
    config.opt_mix = 'momentum'
    config.max_epoch = 100
    config.print()

    logdir = wb.mklogdir('train%d/crf/' % train_num + str(config), is_recreate=True)

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.tag'))
    data.write_file(data.datas[1], os.path.join(logdir, 'valid.id'))
    data.write_file(data.datas[2], os.path.join(logdir, 'test.id'))

    m = crf.CRF(config, data, logdir)

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            ops = crf.DefaultOps(m, data.datas[-1])
            m.train(0.1, ops)


if __name__ == '__main__':
    main()
