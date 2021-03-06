import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from hrf import *
from hrf import trfx_app as trf

from eval import Operation, get_config_rnn, get_config_cnn


def get_config(data):
    config = trf.Config(data)

    # config.word_config = None
    # config.word_config = pot.FeatConfig()
    # config.word_config.feat_dict = {'w[1:4]': 0}
    config.word_config = WordNetConfig(data)
    config.word_config.update(get_config_rnn(data.get_vocab_size()))

    config.chain_num = 100
    config.multiple_trial = 5
    config.jump_width = 1
    config.sample_sub = 5

    config.lr_tag = lr.LearningRateTime(1, 1, tc=10)
    config.lr_mix = lr.LearningRateTime(1, 1, tc=1e4)
    config.lr_word = lr.LearningRateTime(1, 1, tc=1e4)
    config.lr_logz = lr.LearningRateTime(1, 0.2)
    config.opt_tag = 'adam'
    config.opt_mix = 'adam'
    config.opt_word = 'adam'
    config.max_epoch = 1000
    config.train_batch_size = 1000
    config.sample_batch_size = 100
    config.semi_supervised = True

    config.load_crf_model = 'crf/crf_t2g_mixnet1g/trf.mod'

    return config


def main():
    with open('data.info') as f:
        data_info = json.load(f)

    data = seq.DataX(total_level=2,
                     vocab_files=[data_info['hkust_vocab_chr'], data_info['hkust_vocab_pos']],
                     train_list=data_info['hkust_train'],
                     valid_list=data_info['hkust_valid'],
                     )

    config = get_config(data)
    logdir = wb.mklogdir('hrf/' + str(config), is_recreate=True)
    config.print()

    # config.word_config.load_embedding_path = os.path.join(logdir, 'word_emb.txt')
    # if config.word_config.load_embedding_path is not None:
    #     # get embedding vectors
    #     data.create_data().word2vec(config.word_config.load_embedding_path, config.word_config.embedding_dim, cnum=0)

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.tag'))
    data.write_file(data.datas[1], os.path.join(logdir, 'valid.id'))
    # data.write_file(data.datas[2], os.path.join(logdir, 'test.id'))

    m = trf.TRF(config, data, logdir, device='/gpu:0')
    op = Operation(m, data)

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.train(0.1, operation=op)


if __name__ == '__main__':
    main()
