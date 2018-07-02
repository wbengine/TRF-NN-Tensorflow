import tensorflow as tf
import os
import sys
import numpy as np

from base import *
from lm import *
from trf.sa import *
from trf.common import feat
import task


def create_config(data):
    config = trf.Config(data)

    config.chain_num = 100
    config.multiple_trial = 10
    # config.auxiliary_model = 'lstm'
    config.auxiliary_config.embedding_size = 200
    config.auxiliary_config.hidden_size = 200
    config.auxiliary_config.hidden_layers = 1
    config.auxiliary_config.batch_size = 100
    config.auxiliary_config.step_size = 10
    config.auxiliary_config.learning_rate = 1.0

    config.feat_config.feat_type_file = '../../../tfcode/feat/g3.fs'
    config.feat_config.L2_reg = 1e-5
    config.feat_config.pre_compute_data_var = True

    config.net_config = None

    config.lr_feat = lr.LearningRateEpochDelay(1e-3)
    config.lr_net = lr.LearningRateEpochDelay(1e-3)
    config.lr_logz = lr.LearningRateEpochDelay(0.1)
    config.opt_feat_method = 'sgd'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'sgd'

    return config


def create_name(config):
    return str(config) + '_var'


def load_features(data, ftypes):
    info = wb.TxtInfo(task.train)

    # add all the character-ngram
    max_order = 0
    char_ngrams = []
    for w in info.vocab:
        cseq = task.word_seq_to_char([w])
        if not cseq:
            continue

        try:
            cids = data.texts_to_id(cseq)
        except KeyError:
            print('skip w={} c={}'.format(w, cseq))
            continue

        max_order = max(max_order, len(cids))
        char_ngrams.append(cids)

    # add to feat
    print('max_order=', max_order)
    print('ngram_num=', len(char_ngrams))
    single_ftype = feat.SingleFeat('w[1:{}]'.format(max_order))
    single_ftype.add_ngram(char_ngrams, ftypes.num)
    ftypes.insert_single_feat(single_ftype)


def main(_):

    data = task.get_char_data()

    config = create_config(data)
    name = create_name(config)
    logdir = wb.mkdir('trf_sa/' + name, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(logdir, 'trf.log'))
    print(logdir)
    config.print()

    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[0], logdir + '/train.id')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    m = trf.TRF(config, data, logdir=logdir, device='/gpu:0')
    # load_features(data, m.phi_feat.wfeat)

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                             global_step=m.global_step)
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            # print(m.true_logz())
            #
            # m.test_sample()

            m.train(operation=trf.DefaultOps(m, task.NBest()))

if __name__ == '__main__':
    # test_net()
    tf.app.run(main=main)
