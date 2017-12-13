import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from lm import *
from trf.sa import *
import task


class Opt(trf.DefaultOps):
    def __init__(self, trf_model):
        super().__init__(trf_model, *task.get_nbest())
        self.per_epoch = 0.1
        self.next_epoch = 0
        self.out_logz = os.path.join(trf_model.logdir, 'logz.dbg')

    def run(self, step, epoch):
        super().run(step, epoch)

        if epoch > self.next_epoch:
            self.next_epoch += self.per_epoch

            with self.m.time_recoder.recode('true_logz'):
                true_logz = self.m.true_logz(5)
                nce_logz = self.m.norm_const.get_logz()

            with open(self.out_logz, 'at') as f:
                f.write('step={} epoch={:.2f}'.format(step, epoch) + '\n')
                f.write('nce=  ' + ' '.join(['{:.2f}'.format(i) for i in nce_logz]) + '\n')
                f.write('true= ' + ' '.join(['{:.2f}'.format(i) for i in true_logz]) + '\n')


def create_config(data):
    config = trf.Config(data)

    config.chain_num = 100
    config.multiple_trial = 10
    config.sample_batch_size = 100
    # config.auxiliary_model = 'lstm'
    config.auxiliary_config.embedding_size = 32
    config.auxiliary_config.hidden_size = 32
    config.auxiliary_config.hidden_layers = 1
    config.auxiliary_config.batch_size = 100
    config.auxiliary_config.step_size = 10
    config.auxiliary_config.learning_rate = 1.0

    config.lr_feat = lr.LearningRateEpochDelay(1e-3)
    config.lr_net = lr.LearningRateEpochDelay(1e-3)
    config.lr_logz = lr.LearningRateEpochDelay(0.1)
    config.opt_feat_method = 'adam'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'sgd'

    # feat config
    # config.feat_config.feat_type_file = '../../tfcode/feat/g4.fs'
    config.feat_config = None

    # neural config
    config.net_config.update(task.get_config_rnn(config.vocab_size))
    config.net_config.cnn_skip_connection = False

    return config


def create_name(config):
    return str(config)


def main(_):
    data = reader.Data().load_raw_data(reader.word_raw_dir(),
                                       add_beg_token='</s>', add_end_token='</s>',
                                       add_unknwon_token=None,
                                       max_length=None)

    # create config
    config = create_config(data)
    # config.net_config.only_train_weight = True
    # create log dir
    logdir = 'trf_sa/' + create_name(config)
    # prepare the log dir
    wb.prepare_log_dir(logdir, 'trf.log')

    config.print()
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')
    nce_pretrain_model_path = 'trf_nce/trf_nce10_e16_cnn_(1to5)x16_(3x16)x3_relu_noise2gram/trf.mod'

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                             global_step=m.global_step)
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():

            # m.restore_nce_model(nce_pretrain_model_path)
            # m.save()

            m.train(operation=Opt(m))

if __name__ == '__main__':
    tf.app.run(main=main)
