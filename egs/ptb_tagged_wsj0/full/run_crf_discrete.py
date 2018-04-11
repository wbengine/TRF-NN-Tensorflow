import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from hrf import crf


def main():
    with open('data.info') as f:
        data_info = json.load(f)

    data = seq.Data(vocab_files=None,
                    train_list=data_info['train'],
                    valid_list=data_info['valid'],
                    test_list=data_info['test'],
                    )

    config = crf.Config(data)

    # features
    config.tag_config.feat_dict = {'c[1:2]': 0}
    config.mix_config.feat_dict = {'wc[1]': 0,
                                        'w[1]c[1]': 0,
                                        'c[1]w[1]': 0
                                   }

    config.lr_tag = lr.LearningRateTime(1, 1, tc=10)
    config.lr_mix = lr.LearningRateTime(1, 1, tc=10)
    config.opt_tag = 'adam'
    config.opt_mix = 'adam'
    config.max_epoch = 5
    config.print()

    logdir = wb.mklogdir('crf/' + str(config), is_recreate=True)

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.tag'))
    data.write_file(data.datas[1], os.path.join(logdir, 'valid.id'))
    data.write_file(data.datas[2], os.path.join(logdir, 'test.id'))

    m = crf.CRF(config, data, logdir)

    ops = crf.DefaultOps(m, data.datas[-1])

    # m.initialize()
    # with wb.processing('update...'):
    #     m.update(data.datas[0][0:1000])

    # print(m.phi_mix_feat.time_recoder.items())

    # print('cmp wer')
    # ops.run(1, 1)
    #
    # m.update(data.datas[0][0: 100])
    # print(list(m.time_recoder.items()))
    m.train(0.1, ops)


if __name__ == '__main__':
    main()
