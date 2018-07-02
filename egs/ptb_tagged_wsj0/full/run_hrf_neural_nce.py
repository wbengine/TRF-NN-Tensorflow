import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from hrf import *
from hrf import trfnce_semi as trf
from hrf import crf
from eval import get_config_rnn, get_config_cnn


def get_config(data):
    config = trf.Config(data)
    config.word_config.update(get_config_rnn(config.word_vocab_size))

    # config.data_factor = 0.5

    config.lr_tag = lr.LearningRateTime(1e-3)
    config.lr_mix = lr.LearningRateTime(1e-3)
    config.lr_word = lr.LearningRateTime(1e-3)
    config.lr_logz = lr.LearningRateTime(1e-3)
    config.opt_tag = 'adam'
    config.opt_mix = 'adam'
    config.opt_word = 'adam'
    config.opt_logz = 'adam'
    config.max_epoch = 1000

    config.inter_alpha = 1

    # config.prior_model_path = 'lstm/lstm_e200_h200x2/model.ckpt'

    # config.load_crf_model = 'train100/crf/crf_t2g_mixnet1g/trf.mod'
    config.write_dbg = False

    return config


def main():
    with open('data.info') as f:
        data_info = json.load(f)

    data = seq.Data(vocab_files=data_info['vocab'],
                    train_list=data_info['train'],
                    valid_list=data_info['valid'],
                    test_list=data_info['test'],
                    )

    # data.vocabs[0].write('test_vocab.wod')
    # data.vocabs[0].read('test_vocab.wod')
    # data.vocabs[0].write('test_vocab2.wod')
    # return

    data_full = seq.Data(vocab_files=data_info['vocab'],
                         train_list=data_info['train'],
                         valid_list=data_info['valid'],
                         test_list=data_info['test']
                         )

    nbest_files = data_info['nbest']

    config = get_config(data)
    logdir = wb.mklogdir('hrf/' + str(config), is_recreate=True)
    config.print()

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.tag'))
    data.write_file(data.datas[1], os.path.join(logdir, 'valid.id'))
    data.write_file(data.datas[2], os.path.join(logdir, 'test.id'))

    m = trf.TRF(config, data, data_full, logdir, device='/gpu:1')

    # ops = trf.DefaultOps(m, nbest_files, data.datas[-1])
    # ops.nbest_cmp.write_nbest_list(os.path.join(logdir, 'nbest.id'), data)
    ops = crf.DefaultOps(m, data.datas[1], data.datas[2])
    ops.perform_next_epoch = 0
    ops.perform_per_epoch = 1

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():

            m.train(0.1, ops)


if __name__ == '__main__':
    main()
