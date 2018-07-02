import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from hrf import *


def main():
    with open('data.info') as f:
        data_info = json.load(f)

    data = seq.DataX(total_level=2,
                     vocab_files=[data_info['hkust_vocab_chr'], data_info['hkust_vocab_pos']],
                     train_list=data_info['pku_train'],
                     valid_list=data_info['pku_valid'],
                     test_list=data_info['pku_test']
                     )

    config = crf.Config(data)
    config.train_batch_size = 100

    config.lr_tag = lr.LearningRateEpochDelay(0.1)
    config.lr_mix = lr.LearningRateEpochDelay(1e-3)
    config.opt_tag = 'adam'
    config.opt_mix = 'adam'
    config.max_epoch = 10
    config.print()

    logdir = wb.mklogdir('crf/' + str(config), is_recreate=True)

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
