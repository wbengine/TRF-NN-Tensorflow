import tensorflow as tf
import time
import sys
import os
import numpy as np

from base import *
from lm import *


def small_config(data):
    config = lstmlm.Config()
    config.vocab_size = data.get_vocab_size()
    # config.softmax_type = 'Shortlist'
    # config.adaptive_softmax_cutoff = [5000, config.vocab_size]
    config.embedding_size = 200
    config.hidden_size = 200
    config.hidden_layers = 2
    config.step_size = 20
    config.batch_size = 20
    config.epoch_num = 13
    config.init_weight = 0.1
    config.max_grad_norm = 5
    config.dropout = 0
    config.learning_rate = 1.0
    config.lr_decay = 0.5
    config.lr_decay_when = 4
    return config


def medium_config(data):
    config = lstmlm.Config()
    config.vocab_size = data.get_vocab_size()
    config.embedding_size = 650
    config.hidden_size = 650
    config.hidden_layers = 2
    config.step_size = 35
    config.batch_size = 20
    config.epoch_num = 39
    config.init_weight = 0.05
    config.max_grad_norm = 5
    config.dropout = 0.5
    config.learning_rate = 1.0
    config.lr_decay = 0.8
    config.lr_decay_when = 6
    return config


def large_config(data):
    config = lstmlm.Config()
    config.vocab_size = data.get_vocab_size()
    config.embedding_size = 1500
    config.hidden_size = 1500
    config.hidden_layers = 2
    config.step_size = 35
    config.batch_size = 20
    config.epoch_num = 55
    config.init_weight = 0.04
    config.max_grad_norm = 10
    config.dropout = 0.65
    config.learning_rate = 1.0
    config.lr_decay = 1 / 1.15
    config.lr_decay_when = 14
    return config


def create_name(config):
    s = str(config)
    return s


def main(_):
    data = reader.Data().load_raw_data(reader.ptb_raw_dir(),
                                       add_beg_token=None,
                                       add_end_token='</s>',
                                       add_unknwon_token='<unk>')
    nbest = reader.NBest(*reader.wsj0_nbest())
    nbest_list = data.load_data(reader.wsj0_nbest()[0], is_nbest=True)

    config = small_config(data)
    # config = medium_config(data)
    # config.softmax_type = 'AdaptiveSoftmax'
    # config.adaptive_softmax_cutoff = [4000, 8000, config.vocab_size]
    # config = large_config(data)

    work_dir = './lstm/' + create_name(config)
    wb.prepare_log_dir(work_dir, 'lstm.log')
    config.print()

    data.write_vocab(work_dir + '/vocab.txt')
    data.write_data(data.datas[0], work_dir + '/train.id')
    data.write_data(data.datas[1], work_dir + '/valid.id')
    data.write_data(data.datas[2], work_dir + '/test.id')

    write_model = os.path.join(work_dir, 'model.ckpt')  # lm = lstmlm.FastLM(config, device_list=['/gpu:0', '/gpu:0'])

    # lm = lstmlm.LM(config, data, device='/gpu:0')
    lm = lstmlm.FastLM(config, data, device_list=['/gpu:0', '/gpu:1'])

    sv = tf.train.Supervisor(logdir=os.path.join(work_dir, 'logs'),
                             summary_op=None, global_step=lm.global_step())
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:

        ops = lstmlm.DefaultOps(session, lm, nbest, logdir=work_dir)
        # ops.perform(0, 0)

        lm.train(session, data,
                 write_model=write_model,
                 write_to_res=('results.txt', create_name(config)),
                 is_shuffle=False,
                 operation=ops)

        print('compute the WER...')
        t_beg = time.time()
        nbest.lmscore = lm.rescore(session, nbest_list)
        print('rescore time = {:.3f}m'.format((time.time() - t_beg) / 60))
        wb.WriteScore(work_dir + '/nbest.reset.lmscore', nbest.lmscore)
        wer = nbest.wer()
        print('wer={:.3f} lmscale={:.3f}'.format(wer, nbest.lmscale))

        fres = wb.FRes('results.txt')
        fres.AddWER(create_name(config), wer)

if __name__ == '__main__':
    tf.app.run(main=main)
