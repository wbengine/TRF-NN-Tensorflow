import json
import os
import tensorflow as tf
from tqdm import tqdm
import sys

from base import *
from hrf import trfnce as trf, pot, mixphi, tagphi


def main():
    data = seq.Data(vocab_files=None,
                    train_list=[('data/train.chr', 'data/train.tag')],
                    valid_list=[('data/valid.chr', 'data/valid.tag')],
                    unk_token=None,
                    len_count=None,
                    )
    nbest_files = ('data/nbest.words', 'data/transcript.words')

    config = trf.Config(data)

    config.noise_factor = 1
    config.data_factor = 1
    config.sampler_config.embedding_size = 16
    config.sampler_config.hidden_layers = 1
    config.sampler_config.hidden_size = 16

    # features
    config.tag_config.feat_dict = {'c[1:2]': 0}

    config.mix_config = mixphi.MixNetConfig(data)
    config.mix_config.embedding_size = 10
    config.mix_config.hidden_size = 10
    config.mix_config.hidden_layers = 1

    config.word_config = pot.FeatConfig()
    config.word_config.feat_dict = {'w[1:5]': 0}
    # config.word_config.pre_compute_data_exp = True

    config.lr_word = lr.LearningRateTime(0.01)
    config.lr_tag = lr.LearningRateTime(0.01)
    config.lr_mix = lr.LearningRateTime(0.01)
    config.lr_logz = lr.LearningRateTime(0.01)
    config.opt_word = 'adam'
    config.opt_tag = 'adam'
    config.opt_mix = 'adam'
    config.opt_logz = 'adam'

    config.norm_type = 'linear'
    # config.init_logz = config.get_initial_logz(12)

    # config.load_crf_model = 'crf/crf_t2g_mix2g/trf.mod'
    # config.fix_crf_model = True

    logdir = wb.mkdir('hrf/' + str(config), is_recreate=True, force=True)
    sys.stdout = wb.std_log(os.path.join(logdir, 'trf.log'))
    config.print()

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.tag'))

    m = trf.TRF(config, data, logdir)

    ops = trf.DefaultOps(m, nbest_files, data.datas[-1])
    ops.nbest_cmp.write_nbest_list(os.path.join(logdir, 'nbest.id'), data)

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'log'))
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:

        with session.as_default():
            m.train(0.1, ops)


if __name__ == '__main__':
    main()
