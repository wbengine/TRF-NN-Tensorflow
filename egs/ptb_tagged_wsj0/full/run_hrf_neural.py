import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from hrf import *
from hrf import trfx as trf


def main():
    with open('data.info') as f:
        data_info = json.load(f)

    data = seq.Data(vocab_files=None,
                    train_list=data_info['train'],
                    valid_list=data_info['valid'],
                    test_list=data_info['test'],
                    )
    nbest_files = data_info['nbest']

    config = trf.Config(data)

    # features
    config.tag_config.feat_dict = {'c[1:2]': 0}

    config.mix_config = MixNetConfig(data)
    config.mix_config.embedding_size = 200
    config.mix_config.hidden_size = 200
    config.mix_config.hidden_layers = 1
    config.mix_config.max_update_batch = 1000

    # config.word_config = pot.FeatConfig()
    # config.word_config.feat_dict = {'w[1:4]': 0}
    config.word_config = WordNetConfig(data)
    config.word_config.cnn_filters = [(i, 128) for i in range(1, 11)]

    # config.chain_num = 10
    # config.multiple_trial = 10
    # config.jump_width = 2
    # config.sample_sub = 5

    config.lr_tag = lr.LearningRateTime(1, 1, tc=10)
    config.lr_mix = lr.LearningRateTime(1, 1, tc=1e4)
    config.lr_word = lr.LearningRateTime(1, 1, tc=1e4)
    config.lr_logz = lr.LearningRateTime(1, 0.2)
    config.opt_tag = 'adam'
    config.opt_mix = 'adam'
    config.opt_word = 'adam'
    config.max_epoch = 1000

    config.load_crf_model = 'crf/crf_t2g_mixnet1g/trf.mod'
    # config.fix_crf_model = True

    logdir = wb.mklogdir('hrf/' + str(config), is_recreate=True)
    config.print()

    config.word_config.load_embedding_path = os.path.join(logdir, 'word_emb.txt')
    if config.word_config.load_embedding_path is not None:
        # get embedding vectors
        data.create_data().word2vec(config.word_config.load_embedding_path, config.word_config.embedding_dim, cnum=0)

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.tag'))
    data.write_file(data.datas[1], os.path.join(logdir, 'valid.id'))
    data.write_file(data.datas[2], os.path.join(logdir, 'test.id'))

    m = trf.TRF(config, data, logdir, device='/gpu:0')

    ops = trf.DefaultOps(m, nbest_files, data.datas[-1])
    ops.nbest_cmp.write_nbest_list(os.path.join(logdir, 'nbest.id'), data)

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.train(0.1, ops)


if __name__ == '__main__':
    main()
