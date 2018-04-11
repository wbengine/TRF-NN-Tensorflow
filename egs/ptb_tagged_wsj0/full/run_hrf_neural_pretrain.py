import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from trf.sa import trf
from trf.common import net


def get_config_cnn(vocab_size):
    config = net.Config(vocab_size)
    config.embedding_dim = 256
    config.structure_type = 'cnn'
    config.cnn_filters = [(i, 128) for i in range(1, 11)]
    config.cnn_hidden = 128
    config.cnn_width = 3
    config.cnn_layers = 3
    config.cnn_skip_connection = True
    config.cnn_residual = False
    config.cnn_activation = 'relu'
    config.cnn_batch_normalize = False
    return config


def create_config(data):
    config = trf.Config(data)

    config.chain_num = 10
    config.multiple_trial = 10
    config.jump_width = 2
    config.sample_sub = 5
    config.train_batch_size = 1000
    config.sample_batch_size = 100
    config.auxiliary_type = 'lstm'
    config.auxiliary_config.embedding_size = 200
    config.auxiliary_config.hidden_size = 200
    config.auxiliary_config.hidden_layers = 1
    config.auxiliary_config.batch_size = 20
    config.auxiliary_config.step_size = 20
    config.auxiliary_config.learning_rate = 1.0

    config.lr_feat = lr.LearningRateEpochDelay(1e-3)
    config.lr_net = lr.LearningRateTime(1, 1, tc=1e4)
    config.lr_logz = lr.LearningRateTime(1.0, 0.2)
    config.opt_feat_method = 'adam'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'sgd'
    config.max_epoch = 1000

    # feat config
    # config.feat_config.feat_type_file = '../../tfcode/feat/g4.fs'
    config.feat_config = None

    # neural config
    config.net_config.update(get_config_cnn(config.vocab_size))

    return config


def main():
    with open('data.info') as f:
        data_info = json.load(f)

    data_all = seq.Data(vocab_files=None,
                        train_list=data_info['train'],
                        valid_list=data_info['valid'],
                        test_list=data_info['test'],
                        )
    data = data_all.create_data()
    nbest_files = data_info['nbest']

    config = create_config(data)

    logdir = wb.mklogdir('hrf_pretrain/' + str(config), is_recreate=True)
    config.print()

    config.net_config.load_embedding_path = os.path.join(logdir, 'word_emb.txt')
    if config.net_config.load_embedding_path is not None:
        # get embedding vectors
        data.word2vec(config.net_config.load_embedding_path, config.net_config.embedding_dim, cnum=0)

    config.print()
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')

    ops = trf.DefaultOps(m, nbest_files)
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
