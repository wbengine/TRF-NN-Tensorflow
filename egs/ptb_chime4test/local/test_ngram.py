import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from trf.nce import *
import task


def create_name(config):
    return 'test_ngram'


def main(_):
    data = reader.Data().load_raw_data([task.train, task.valid, task.test],
                                       add_beg_token='</s>', add_end_token='</s>'
                                       )

    config = trf.Config(data)
    config.max_epoch = 100
    config.batch_size = 20
    config.noise_factor = 10
    config.noise_sampler = '1gram'
    # config.init_logz = config.get_initial_logz(0)
    config.lr_feat = lr.LearningRateEpochDelay(1.0)
    config.lr_net = lr.LearningRateEpochDelay(1e-3)
    config.lr_logz = lr.LearningRateEpochDelay(1e-3)
    config.opt_feat_method = 'sgd'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'sgd'

    # config.prior_model_path = 'lstm/lstm_e32_h32x1_BNCE_SGD/model.ckpt'
    # feat config
    # config.feat_config.feat_type_file = '../../tfcode/feat/g3.fs'
    # config.feat_config.feat_cluster = None
    config.feat_config = None

    # net config
    config.net_config.structure_type = 'rnn'
    config.net_config.rnn_type = 'lstm'
    config.net_config.embedding_dim = 200
    config.net_config.rnn_hidden_size = 200
    config.net_config.rnn_hidden_layers = 2
    config.net_config.rnn_predict = True

    name = create_name(config)
    logdir = 'trf_nce_new/' + name
    wb.mkdir(logdir, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(logdir, 'trf.log'))
    print(logdir)
    config.print()

    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    # # 1gram:
    # valid_wer = 11.819908184147938
    # test_wer = 14.124927028604786
    # lmscale = 9

    # 2gram:
    # valid_wer = 10.282269215877873
    # test_wer = 12.870986573263279
    # lmscale = 8

    # 3gram:
    # valid_wer= 11.016058555651838
    # test_wer= 13.046117921774664
    # lmscale= 8

    # wb.rmdir(logdirs)
    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')

    ops = task.Ops(m)
    nbest_cmp = ops.nbest_cmp
    noise_sampler = m.noise_sampler

    for nbest in nbest_cmp.nbests:
        nbest.lmscore = -noise_sampler.noise_logps(nbest.get_nbest_list(data))
    nbest_cmp.cmp_wer()

    print('valid_wer=', nbest_cmp.get_valid_wer())
    print('test_wer=', nbest_cmp.get_test_wer())
    print('lmscale=', nbest_cmp.lmscale)

    return

    # print('noise_nll=', -np.mean(m.noise_sampler.noise_logps(data.datas[0])))
    # return

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                             global_step=m.global_step)
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:

        with session.as_default():

            # print(m.eval(data.datas[1]))

            m.train(print_per_epoch=0.1,
                    operation=task.Ops(m))


if __name__ == '__main__':
    tf.app.run(main=main)
