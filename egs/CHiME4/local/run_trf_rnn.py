import tensorflow as tf
import sys
import os
import numpy as np
import time

import task

from model import reader
from model import trfnnbase as trf
from model import wblib as wb

# [data]
data = reader.Data().load_raw_data([task.train, task.valid, task.valid], add_beg_token='<s>', add_end_token='</s>')
data.cut_train_to_length(50)
nbest_cmp = task.NBestComputer()


class Ops(trf.Operation):
    def __init__(self, trf_model):
        super().__init__(trf_model)
        self.wer_next_epoch = 0
        self.opt_det_wer = 100
        self.opt_txt_wer = 100
        self.write_models = wb.mkdir(os.path.join(self.m.logdir, 'wer_models'))

    def run(self, step, epoch):
        super().run(step, epoch)

        if epoch >= self.wer_next_epoch:
            self.wer_next_epoch = int(epoch) + 1

            print('[Ops] rescoring:', end=' ', flush=True)

            # resocring
            with self.m.time_recoder.recode('rescore'):
                time_beg = time.time()
                for nbest in nbest_cmp.nbests:
                    nbest.lmscore = -self.m.get_log_probs(nbest.get_nbest_list(self.m.data))
                rescore_time = time.time() - time_beg
            # compute wer
            with self.m.time_recoder.recode('wer'):
                time_beg = time.time()
                nbest_cmp.cmp_wer()
                nbest_cmp.write_to_res(os.path.join(self.m.logdir, 'wer_per_epoch.log'), 'epoch%d' % int(epoch))
                dev_wer = nbest_cmp.get_valid_wer()
                tst_wer = nbest_cmp.get_test_wer()
                wer_time = time.time() - time_beg
                print('epoch={:.2f} dev_wer={:.2f} test_wer={:.2f} lmscale={} '
                      'rescore_time={:.2f}, wer_time={:.2f}'.format(
                    epoch, dev_wer, tst_wer, nbest_cmp.lmscale,
                    rescore_time / 60, wer_time / 60))

            # write models
            if dev_wer < self.opt_det_wer:
                self.opt_det_wer = dev_wer
                self.m.save(self.write_models + '/epoch%d' % int(epoch))


def create_name(config):
    return 'trf_' + str(config.config_trf) + '_maxlen{}'.format(config.max_len)


def get_config():
    config = trf.Config(data, 'rnn')
    config.jump_width = 2
    config.chain_num = 100
    config.batch_size = 100
    config.lr_cnn = trf.trfbase.LearningRateTime(beta=1.0, tc=1e4)
    config.lr_zeta = trf.trfbase.LearningRateTime(1.0, 0.2)
    config.max_epoch = 1000

    config_trf = config.config_trf
    config_trf.opt_method = 'adam'
    config_trf.embedding_dim = 200
    config_trf.hidden_layers = 2
    config_trf.hidden_dim = 200
    config_trf.init_weight = 0.1
    config_trf.max_grad_norm = 10
    config_trf.zeta_gap = 10
    config_trf.train_batch_size = 1000
    config_trf.sample_batch_size = 400
    config_trf.update_batch_size = 500
    config_trf.dropout = 0

    config_lstm = config.config_lstm
    config_lstm.hidden_size = 200
    config_lstm.step_size = 100

    return config


def main(_):

    config = get_config()
    name = create_name(config)
    logdir = wb.mkdir('./trf_nn/' + name, is_recreate=False)
    sys.stdout = wb.std_log(logdir + '/trf.log')
    config.pprint()
    print(logdir)

    # write data
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[0], logdir + '/train.id')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    m = trf.TRF(config, data, logdir=logdir, device=['/gpu:0', '/gpu:1'])
    ops = Ops(m)

    sv = tf.train.Supervisor(logdir=logdir + '/logs', summary_op=None, global_step=m.global_steps)
    # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        m.set_session(session)

        m.pre_train(sv, session, batch_size=100, max_epoch=10, lr=1e-3)

        m.train(sv, session,
                print_per_epoch=0.01,
                operation=ops,
                model_per_epoch=None)


if __name__ == '__main__':
    # pretrain()
    tf.app.run(main=main)
