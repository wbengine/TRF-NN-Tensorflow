import tensorflow as tf
import time
import sys
import os
import numpy as np

from lm import *
from base import *


def small_config(data):
    config = lstmlm.Config()
    config.vocab_size = data.get_vocab_size()
    config.embedding_size = 32
    config.hidden_size = 32
    config.hidden_layers = 1
    config.step_size = 5
    config.batch_size = 100
    config.epoch_num = 10
    config.init_weight = 0.1
    config.max_grad_norm = 5
    config.dropout = 0
    config.learning_rate = 1.
    config.lr_decay = 0.5
    config.lr_decay_when = 4
    return config


def create_name(config):
    s = 'lstm_{}x{}'.format(config.hidden_size, config.hidden_layers)
    return s


def main(_):
    data = reader.Data().load_raw_data(reader.word_raw_dir(),
                                       add_beg_token=None,
                                       add_end_token='</s>',
                                       add_unknwon_token=None)
    nbest = reader.NBest(*reader.word_nbest())
    nbest_list = data.load_data(nbest.nbest, is_nbest=True)

    config = small_config(data)

    work_dir = './lstm/' + create_name(config)
    wb.mkdir(work_dir, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(work_dir, 'lstm.log'))
    print(work_dir)
    wb.pprint_dict(config.__dict__)

    data.write_vocab(work_dir + '/vocab.txt')
    data.write_data(data.datas[0], work_dir + '/train.id')
    data.write_data(data.datas[1], work_dir + '/valid.id')
    data.write_data(data.datas[2], work_dir + '/test.id')
    data.write_data(nbest_list, work_dir + '/nbest.id')

    write_model = os.path.join(work_dir, 'model.ckpt')

    lm = lstmlm.LM(config, device='/gpu:0')

    sv = tf.train.Supervisor(logdir=os.path.join(work_dir, 'logs'),
                             summary_op=None, global_step=lm.global_step())
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:

        lm.train(session, data, write_model, ('data/models_ppl.txt', create_name(config)))

        print('compute the WER...')
        t_beg = time.time()
        nbest.lmscore = lm.rescore(session, nbest_list)
        print('rescore time = {:.3f}m'.format((time.time() - t_beg) / 60))
        wb.WriteScore('nbest.reset.lmscore', nbest.lmscore)
        wer = nbest.wer()
        print('wer={:.3f} lmscale={:.3f}'.format(wer, nbest.lmscale))

        fres = wb.FRes('data/models_ppl.txt')
        fres.AddWER(create_name(config), wer)

if __name__ == '__main__':
    tf.app.run(main=main)
