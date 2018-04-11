import json
import sys
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from trf.common import seq
from hrf import *


def main():
    data = seq.Data(vocab_files=None,
                    train_list=[('data/train.chr', 'data/train.tag')],
                    valid_list=[('data/valid.chr', 'data/valid.tag')],
                    unk_token=None,
                    len_count=None,
                    )
    nbest_files = ('data/nbest.words', 'data/transcript.words')

    config = crf.Config(data)

    # features
    config.tag_config = TagConfig(data)
    config.tag_config.feat_dict = {'c[1:2]': 0}

    config.mix_config = MixNetConfig(data)
    config.mix_config.embedding_size = 10
    config.mix_config.hidden_size = 10
    config.mix_config.hidden_layers = 1

    config.lr_tag = lr.LearningRateEpochDelay(1.0)
    config.lr_mix = lr.LearningRateEpochDelay(0.1)
    config.opt_tag = 'sgd'
    config.opt_mix = 'sgd'
    config.max_epoch = 1
    config.print()

    logdir = wb.mkdir('crf/' + str(config), is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(logdir, 'trf.log'))

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.tag'))
    data.write_file(data.datas[1], os.path.join(logdir, 'valid.id'))

    m = crf.CRF(config, data, logdir)
    ops = crf.DefaultOps(m, data.datas[-1])

    sv = tf.train.Supervisor()
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.train(0.1, ops)


if __name__ == '__main__':
    main()
