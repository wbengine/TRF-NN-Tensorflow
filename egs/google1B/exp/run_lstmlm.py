import tensorflow as tf
import time
import sys
import os
import glob
import numpy as np
import json

from base import *
from lm import *
from eval import OperationLSTM


def small_config(data):
    config = lstmlm.Config()
    config.vocab_size = data.get_vocab_size()
    config.softmax_type = 'AdaptiveSoftmax'
    config.adaptive_softmax_cutoff = [10000, 50000, 100000, data.get_vocab_size()]
    # config.softmax_type = 'Shortlist'
    # config.adaptive_softmax_cutoff = [10000, data.get_vocab_size()]
    config.embedding_size = 256
    config.hidden_size = 1024
    config.hidden_layers = 2
    config.step_size = 20
    config.batch_size = 32
    config.epoch_num = 5
    config.init_weight = 0.05
    config.max_grad_norm = 5
    config.dropout = 0.5
    config.optimize_method = 'adam'
    config.learning_rate = 1e-3
    config.lr_decay = 0.5
    config.lr_decay_when = 1
    return config

res_file = 'results.txt'
fres = wb.FRes(res_file)  # the result file

with open('../data.info') as f:
    data_info = json.load(f)


def main(_):
    train_files = 100
    data = reader.LargeData().dynamicly_load_raw_data(sorted_vocab_file=data_info['vocab_cut3'],
                                                      train_list=data_info['train_all'][0: train_files],
                                                      valid_file=data_info['valid'],
                                                      test_file=data_info['test'],
                                                      add_beg_token='<s>',
                                                      add_end_token='</s>',
                                                      add_unknwon_token='<unk>',
                                                      vocab_max_size=None)

    nbest = reader.NBest(*reader.wsj0_nbest())

    config = small_config(data)
    # config = medium_config(data)
    # config = large_config(data)
    model_name = 't{}_'.format(train_files) + str(config)
    work_dir = './lstmlm/' + model_name
    wb.mkdir(work_dir, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(work_dir, 'lstm.log'))
    print(work_dir)
    config.print()

    data.write_vocab(work_dir + '/vocab.txt')
    data.write_data(data.datas[1], work_dir + '/valid.id')
    data.write_data(data.datas[2], work_dir + '/test.id')

    write_model = os.path.join(work_dir, 'model.ckpt')

    # lm = lstmlm.LM(config, data, device='/gpu:0')
    lm = lstmlm.FastLM(config, data, device_list=['/gpu:1', '/gpu:0'])

    sv = tf.train.Supervisor(logdir=os.path.join(work_dir, 'logs'),
                             summary_op=None, global_step=lm.global_step())
    # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:

        ops = OperationLSTM(lm, data, work_dir,
                            rescore_fun=lambda x: lm.rescore(session, x, reset_state_for_sentence=True))
        # ops.perform(0, 0)

        lm.train(session, data,
                 write_model=write_model,
                 print_per_epoch=0.01,
                 write_to_res=('results.txt', model_name),
                 operation=ops)

        print('compute the WER...')
        t_beg = time.time()
        nbest.lmscore = lm.rescore(session, nbest.get_nbest_list(data), reset_state_for_sentence=True)
        print('rescore time = {:.3f}m'.format((time.time() - t_beg) / 60))
        wb.WriteScore('nbest.reset.lmscore', nbest.lmscore)
        print('wer={:.3f} lmscale={:.3f}'.format(nbest.wer(), nbest.lmscale))

        fres = wb.FRes('results.txt')
        fres.AddWER(model_name, nbest.wer())


if __name__ == '__main__':
    tf.app.run(main=main)
