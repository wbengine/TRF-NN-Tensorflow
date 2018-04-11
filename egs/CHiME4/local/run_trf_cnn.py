import tensorflow as tf
import sys
import os
import numpy as np
import time

import task

from model import reader
from model import trfcnn as trf
from model import wblib as wb
from model import lstmlm
import run_lstmlm_withBegToken

# [data]
data = reader.Data().load_raw_data([task.train, task.valid, task.valid], add_beg_token='<s>', add_end_token='</s>')
data.cut_train_to_length(50)
nbest_cmp = task.NBestComputer()


class Ops(trf.trfjsa.Operation):
    def __init__(self, trf_model):
        super().__init__(trf_model)
        self.wer_next_epoch = 0
        self.wer_per_epoch = 0.2
        self.opt_det_wer = 100
        self.opt_txt_wer = 100
        self.write_models = wb.mkdir(os.path.join(self.m.logdir, 'wer_models'))

    def run(self, step, epoch):
        super().run(step, epoch)

        if epoch >= self.wer_next_epoch:
            self.wer_next_epoch = (int(epoch / self.wer_per_epoch) + 1) * self.wer_per_epoch
            epoch_num = int(epoch / self.wer_per_epoch) * self.wer_per_epoch

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
                nbest_cmp.write_to_res(os.path.join(self.m.logdir, 'wer_per_epoch.log'), 'epoch%.2f' % epoch_num)
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
                self.m.save(self.write_models + '/epoch%.2f' % epoch_num)


def create_name(config, q_config):
    return str(config) + '_with_' + run_lstmlm_withBegToken.create_name(q_config) + '_zeta0'


def get_config():
    config = trf.Config(data)
    config.embedding_dim = 256
    config.cnn_filters = [(i, 128) for i in range(1, 11)]
    config.cnn_layers = 3
    config.cnn_hidden = 128
    config.max_epoch = 1000
    config.init_weight = 0.05
    config.opt_method = 'adam'
    config.max_grad_norm = None
    config.jump_width = 2
    config.chain_num = 10
    config.train_batch_size = 1000
    config.sample_batch_size = 300
    config.batch_normalize = False
    config.lr_cnn = trf.trfjsa.LearningRateTime(1e-4)
    config.lr_zeta = trf.trfjsa.LearningRateTime(1.0, 0.2)
    config.auxiliary_model = 'lstm'
    config.auxiliary_lr = 1.
    return config


def main(_):

    config = get_config()
    q_config = run_lstmlm_withBegToken.small_config(data)
    name = create_name(config, q_config)
    logdir = wb.mkdir('./trf_nn/' + name, is_recreate=True)
    sys.stdout = wb.std_log(logdir + '/trf.log')
    config.pprint()
    print(logdir)

    # write data
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[0], logdir + '/train.id')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0', simulater_device='/gpu:0',
                q_model=lstmlm.LM(run_lstmlm_withBegToken.small_config(data), device='/gpu:0'))
    ops = Ops(m)

    sv = tf.train.Supervisor(logdir=logdir + '/logs', summary_op=None, global_step=m._global_step)
    # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        m.set_session(session)

        print('load lstmlm q(x)')
        m.q_model.restore(session,
                          './lstm/' + run_lstmlm_withBegToken.create_name(m.q_model.config) + '/model.ckpt')

        m.train(sv, session,
                print_per_epoch=0.05,
                operation=ops,
                model_per_epoch=None)


if __name__ == '__main__':
    # pretrain()
    tf.app.run(main=main)
