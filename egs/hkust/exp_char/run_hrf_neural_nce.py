import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from trf.common import net
import hrf
from hrf import trfnce as trf

from eval import Operation, get_config_rnn


def get_config(data):
    config = trf.Config(data)

    #########################
    # model config
    #########################
    # tag features
    config.tag_config.feat_dict = {'c[1:2]': 0}
    # mix features
    config.mix_config = hrf.MixNetConfig(data)
    config.mix_config.embedding_size = 200
    config.mix_config.hidden_size = 200
    config.mix_config.hidden_layers = 1
    config.mix_config.max_update_batch = 1000
    # word features
    config.word_config = hrf.WordNetConfig(data)
    config.word_config.update(get_config_rnn(config.word_vocab_size))

    ##################################
    # model learning
    ##################################
    # learning rate and opt method
    config.lr_tag = lr.LearningRateTime(1e-3)
    config.lr_mix = lr.LearningRateTime(1e-3)
    config.lr_word = lr.LearningRateTime(1e-3)
    config.lr_logz = lr.LearningRateTime(1e-2)
    config.opt_tag = 'adam'
    config.opt_mix = 'adam'
    config.opt_word = 'adam'
    config.max_epoch = 1000
    # NCE configures
    config.batch_size = 200
    config.noise_factor = 1
    config.data_factor = 1
    config.semi_supervised = True
    config.update_aux_using_samples = True

    config.write_dbg = True

    config.load_crf_model = 'crf/crf_t2g_mixnet1g/trf.mod'
    # config.fix_crf_model = True

    return config

def main():
    with open('data.info') as f:
        data_info = json.load(f)

    data = seq.DataX(total_level=2,
                     vocab_files=[data_info['hkust_vocab_chr'], data_info['hkust_vocab_pos']],
                     train_list=data_info['hkust_train'],
                     valid_list=data_info['hkust_valid'],
                     test_list=data_info['pku_test']
                     )

    config = get_config(data)
    logdir = wb.mklogdir('hrf_nce/' + str(config), is_recreate=True)
    config.print()

    # config.word_config.load_embedding_path = os.path.join(logdir, 'word_emb.txt')
    # if config.word_config.load_embedding_path is not None:
    #     # get embedding vectors
    #     data.create_data().word2vec(config.word_config.load_embedding_path, config.word_config.embedding_dim, cnum=0)

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.pos'))
    data.write_file(data.datas[1], os.path.join(logdir, 'valid.id'))

    m = trf.TRF(config, data, logdir, device='/gpu:1')

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.train(0.1, Operation(m, data))


if __name__ == '__main__':
    main()
