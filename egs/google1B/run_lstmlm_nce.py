import tensorflow as tf
import time
import sys
import os
import glob
import numpy as np
import corpus

from base import *
from lm import *


def small_config(data):
    config = lstmlm.Config()
    config.vocab_size = data.get_vocab_size()
    config.softmax_type = 'BNCE'
    config.fixed_logz_for_nce = 9
    config.embedding_size = 100
    config.hidden_size = 100
    config.hidden_layers = 1
    config.step_size = 20
    config.batch_size = 128
    config.epoch_num = 5
    config.init_weight = 0.05
    config.max_grad_norm = 5
    config.dropout = 0.01
    config.optimize_method = 'adam'
    config.learning_rate = 1e-3
    config.lr_decay = 0.5
    config.lr_decay_when = 1
    return config


def create_name(config):
    s = 'lstm_v{}_e{}_h{}x{}'.format(config.vocab_size, config.embedding_size, config.hidden_size, config.hidden_layers)
    if config.softmax_type != 'Softmax':
        s += '_' + config.softmax_type
    return s


def main(_):
    max_files = 100
    sorted_vocab = './data/vocab_cutoff9.txt'
    reverse_sentence = False

    train_list, valid_file, test_file = corpus.word_raw_dir(max_files)
    data = reader.LargeData().dynamicly_load_raw_data(sorted_vocab,
                                                      train_list, valid_file, test_file,
                                                      add_beg_token=None,
                                                      add_end_token='</s>',
                                                      add_unknwon_token='<unk>',
                                                      reverse_sentence=reverse_sentence)
    nbest = reader.NBest(*reader.wsj0_nbest())
    nbest_list = data.load_data(reader.wsj0_nbest()[0], is_nbest=True, reversed_sentence=reverse_sentence)

    config = small_config(data)
    # config = medium_config(data)
    # config = large_config(data)

    work_dir = './lstmlm/' + create_name(config)
    wb.mkdir(work_dir, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(work_dir, 'lstm.log'))
    config.print()
    print(work_dir)

    data.write_vocab(work_dir + '/vocab.txt')
    data.write_data(data.datas[1], work_dir + '/valid.id')
    data.write_data(data.datas[2], work_dir + '/test.id')
    data.write_data(nbest_list, work_dir + '/nbest.id')

    write_model = os.path.join(work_dir, 'model.ckpt')

    with tf.Graph().as_default():
        lm = lstmlm.LM(config, data, device='/gpu:0')
        # lm = lstmlm.FastLM(config, data, device_list=['/gpu:0', '/gpu:1'])

        sv = tf.train.Supervisor(logdir=os.path.join(work_dir, 'logs'),
                                 summary_op=None, global_step=lm.global_step())
        # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:

            lm.train(session, data,
                     write_model=write_model,
                     write_to_res=('results.txt', create_name(config)))

            print('compute the WER...')
            t_beg = time.time()
            nbest.lmscore = lm.rescore(session, nbest_list, reset_state_for_sentence=True)
            print('rescore time = {:.3f}m'.format((time.time() - t_beg)/60))
            wb.WriteScore('nbest.reset.lmscore', nbest.lmscore)
            print('wer={:.3f} lmscale={:.3f}'.format(nbest.wer(), nbest.lmscale))

            fres = wb.FRes('results.txt')
            fres.AddWER(create_name(config), nbest.wer())


if __name__ == '__main__':
    tf.app.run(main=main)
