import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from hrf import *
from hrf import trfx_semi as trf
from hrf import crf
from eval import get_config_rnn, get_config_cnn


def get_config(data):
    config = trf.Config(data)
    config.word_config = None
    # config.word_config.update(get_config_rnn(config.word_vocab_size))

    config.chain_num = 100
    config.multiple_trial = 5
    config.jump_width = 1
    config.sample_sub = 5

    # config.sample_batch_size = 300

    config.lr_tag = lr.LearningRateTime(tc=1e-3)
    config.lr_mix = lr.LearningRateTime(tc=1e-3)
    config.lr_word = lr.LearningRateTime(1, 1, tc=1e4)
    config.lr_logz = lr.LearningRateTime(1, 0.2)
    config.opt_tag = 'adam'
    config.opt_mix = 'adam'
    config.opt_word = 'adam'
    config.max_epoch = 1000

    # config.prior_model_path = 'lstm/lstm_e200_h200x2/model.ckpt'

    config.semi_supervised = True
    config.load_crf_model = 'train100/crf/crf_t2g_mixnet1g/trf.mod'

    return config


train_num = 100

def main():
    with open('data.info') as f:
        data_info = json.load(f)

    data = seq.Data(vocab_files=data_info['vocab'],
                    train_list=data_info['train'],
                    valid_list=data_info['valid'],
                    test_list=data_info['test'],
                    )

    data_full = seq.Data(vocab_files=data_info['vocab'],
                         train_list=data_info['train%d' % train_num],
                         valid_list=data_info['valid'],
                         test_list=data_info['test']
                         )

    nbest_files = data_info['nbest']

    config = get_config(data)
    logdir = wb.mklogdir('train%d/' % train_num + str(config), is_recreate=True)
    config.print()

    # config.word_config.load_embedding_path = os.path.join(logdir, 'word_emb.txt')
    # if config.word_config.load_embedding_path is not None:
    #     # get embedding vectors
    #     data.create_data().word2vec(config.word_config.load_embedding_path, config.word_config.embedding_dim, cnum=0)

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.tag'))
    data.write_file(data.datas[1], os.path.join(logdir, 'valid.id'))
    data.write_file(data.datas[2], os.path.join(logdir, 'test.id'))

    m = trf.TRF(config, data, data_full, logdir, device='/gpu:1')

    # ops = trf.DefaultOps(m, nbest_files, data.datas[-1])
    # ops.nbest_cmp.write_nbest_list(os.path.join(logdir, 'nbest.id'), data)
    ops = crf.DefaultOps(m, data.datas[-1])
    ops.perform_next_epoch = 0
    # ops.perform_per_epoch = 0.1

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.train(0.1, ops)


if __name__ == '__main__':
    main()
