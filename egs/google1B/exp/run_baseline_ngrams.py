import os
import sys
import json
import numpy as np

from base import *
from lm import ngramlm

if wb.is_window():
    bindir = 'd:\\wangbin\\tools'
else:
    bindir = '../../../tools/srilm/'

res_file = 'results.txt'
fres = wb.FRes(res_file)  # the result file

with open('../data.info') as f:
    data_info = json.load(f)


def main():
    train_files = 100
    data = reader.LargeData().dynamicly_load_raw_data(sorted_vocab_file=data_info['vocab_cut3'],
                                                      train_list=data_info['train_all'][0: train_files],
                                                      valid_file=data_info['valid'],
                                                      test_file=data_info['test'],
                                                      add_beg_token='<s>',
                                                      add_end_token='</s>',
                                                      add_unknwon_token='<unk>',
                                                      vocab_max_size=None)

    nbest = reader.NBest(*reader.wsj0_nbest())

    config = ngramlm.Config(data)
    config.res_file = 'results.txt'

    order_reg = [5]
    for order in order_reg:
        config.order = order
        config.cutoff = [0, 0, 2, 2, 5]

        model_name = 't{}_'.format(train_files) + str(config)
        workdir = 'ngramlm/' + model_name
        sys.stdout = wb.std_log(os.path.join(workdir, 'ngram.log'))
        datadir = 'ngramlm/data/'
        m = ngramlm.Model(config, data, bindir, workdir, datadir, name=model_name)

        print('train...')
        with wb.processing('training'):
            m.train(write_to_res=(res_file, model_name))

        print('rescore...')
        with wb.processing('rescoring'):
            nbest.lmscore = m.rescore(nbest.get_nbest_list(data))
        nbest.write_lmscore(os.path.join(workdir, 'nbest.lmscore'))
        wer = nbest.wer()
        print('wer={} lmscale={}, acscale={}'.format(wer, nbest.lmscale, nbest.acscale))
        fres.AddWER(model_name, wer)

if __name__ == '__main__':
    main()
