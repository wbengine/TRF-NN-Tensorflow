import json
import os
import sys
import tensorflow as tf
from tqdm import tqdm

from base import *
from hrf import *
from hrf import trf_IS as trf


def main():
    with open('data.info') as f:
        data_info = json.load(f)

    data = seq.Data(vocab_files=None,
                    train_list=data_info['train'],
                    valid_list=data_info['valid'],
                    test_list=data_info['test'],
                    max_len=80
                    )
    nbest_files = data_info['nbest']

    config = trf.Config(data)
    config.multiple_trial = 1
    config.chain_num = 100
    config.sample_sub = 1
    config.auxiliary_config.embedding_size = 200
    config.auxiliary_config.hidden_size = 200
    config.auxiliary_config.hidden_layers = 1

    # features
    config.tag_config = tagphi.TagConfig(data)
    config.tag_config.feat_dict = {'c[1:2]': 0}

    config.mix_config = mixphi.MixFeatConfig(data)
    config.mix_config.feat_dict = {'wc[1]': 0,
                                   'w[1]c[1]': 0,
                                   'c[1]w[1]': 0}

    config.word_config = pot.FeatConfig()
    config.word_config.feat_dict = {'w[1:4]': 0}
    # config.word_config = None

    config.lr_word = lr.LearningRateTime(1, 1, tc=1e1)
    config.lr_tag = lr.LearningRateTime(1, 1, tc=1e1)
    config.lr_mix = lr.LearningRateTime(1, 1, tc=1e1)
    config.lr_logz = lr.LearningRateTime(1, 1, tc=1e1)
    config.opt_word = 'adam'
    config.opt_tag = 'adam'
    config.opt_mix = 'adam'
    config.opt_logz = 'sgd'

    # config.load_crf_model = 'crf/crf_t2g_mix2g/trf.mod'

    # config.global_logz = 1000

    logdir = wb.mklogdir('hrf/' + str(config), is_recreate=True)
    config.print()
    print(logdir)

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.tag'))

    m = trf.TRF(config, data, logdir)

    ops = trf.DefaultOps(m, nbest_files, data.datas[-1])
    ops.nbest_cmp.write_nbest_list(os.path.join(logdir, 'nbest.id'), data)

    # m.initialize()
    # m.save()
    #
    # print('true_pi', config.pi_true[10 + 2], config.pi_true[11 + 2])
    # nbest_list = ops.nbest_cmp.get_nbest_list(data)
    # x_list = nbest_list[0:2]
    # t_list, _ = m.get_tag(x_list)
    # s_list = [seq.Seq([x, t]) for x, t in zip(x_list, t_list)]
    # print('x', m.get_logpxs(x_list))
    #
    # ops.wer_next_epoch = 0.0

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'log'))
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:

        with session.as_default():
            m.train(0.1, ops)


if __name__ == '__main__':
    main()
