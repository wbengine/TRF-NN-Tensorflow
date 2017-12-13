import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from trf.nce import *


def create_name(config):
    return str(config)


def main(_):
    data = reader.Data().load_raw_data(reader.word_raw_dir(),
                                       add_beg_token='</s>', add_end_token='</s>',
                                       add_unknwon_token=None,
                                       max_length=None,
                                       )

    config = trf.Config(data)
    config.write_dbg = False
    config.max_epoch = 100
    config.batch_size = 20
    config.noise_factor = 10
    config.noise_sampler = '2gram'
    config.lr_feat = lr.LearningRateEpochDelay(1e-3)
    config.lr_net = lr.LearningRateEpochDelay(1e-3)
    config.lr_logz = lr.LearningRateEpochDelay(1e-3)
    config.opt_feat_method = 'adam'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'adam'

    # config.prior_model_path = 'lstm/lstm_e32_h32x1_BNCE_SGD/model.ckpt'
    # feat config
    config.feat_config.feat_type_file = '../../tfcode/feat/g4.fs'
    config.feat_config.feat_cluster = None

    # net config
    config.net_config = None

    name = create_name(config)
    logdir = 'trf_nce/' + name
    wb.mkdir(logdir, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(logdir, 'trf.log'))
    print(logdir)
    config.print()

    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    # wb.rmdir(logdirs)
    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                             global_step=m.global_step)
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.train(operation=trf.DefaultOps(m, *reader.word_nbest()))


if __name__ == '__main__':
    tf.app.run(main=main)
