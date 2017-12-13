import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from trf.nce import *
import task


def create_name(config):
    return str(config)+"_add_noise"


def main(_):
    data = reader.Data().load_raw_data([task.train, task.valid, task.test],
                                       add_beg_token='</s>', add_end_token='</s>'
                                       )

    config = trf.Config(data)
    config.feat_type_file = '../../../tfcode/feat/g3.fs'
    config.max_epoch = 20
    config.batch_size = 10
    config.noise_factor = 10
    config.noise_sampler = '1gram'
    config.lr_feat = lr.LearningRateEpochDelay(1e-3)
    config.lr_logz = lr.LearningRateEpochDelay(1e-3)
    config.opt_feat_method = 'sgd'
    config.opt_logz_method = 'sgd'

    name = create_name(config)
    logdir = 'trf_nce_new/' + name
    wb.mkdir(logdir, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(logdir, 'trf.log'))
    print(logdir)
    config.print()

    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    # wb.rmdir(logdirs)
    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')

    # logps = m.noise_sampler.noise_logps(data.datas[0])
    # print('noise_sample_nll=', np.mean(logps))

    m.train(print_per_epoch=0.1,
            operation=task.Ops(m))



    # sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
    #                          global_step=m.train_net.global_step)
    # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    # session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # session_config.gpu_options.allow_growth = True
    # with sv.managed_session(config=session_config) as session:
    #
    #     with session.as_default():
    #
    #         # m.noise_sampler.start()
    #         #
    #         # s = [0, 10, 18, 13, 1, 9, 63, 9, 7, 22, 4, 0, 5, 0]
    #         # print(m.noise_sampler.noise_logps([s]))
    #
    #         m.train(sv, session,
    #                 print_per_epoch=0.1,
    #                 nbest=nbest,
    #                 operation=Opt(m))


if __name__ == '__main__':
    tf.app.run(main=main)
