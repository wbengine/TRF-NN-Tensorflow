import os
import sys
import numpy as np
import time
import task

from base import *
from lm import *

if wb.is_window():
    bindir = 'd:\\wangbin\\tools'
else:
    bindir = '../../../tools/srilm/'


def get_name(config):
    return str(config)


def main():
    data = task.get_word_data()
    nbest = task.NBest()

    config = ngramlm.Config(data)

    order_reg = [3, 4, 5]
    for order in order_reg:
        config.order = order
        config.cutoff = [0] * order

        workdir = wb.mkdir('ngramlm/' + get_name(config), is_recreate=False)
        sys.stdout = wb.std_log(workdir + '/ngram.log')
        print(workdir)

        m = ngramlm.Model(config, data, bindir, workdir)

        # train
        with wb.processing('train...'):
            m.train(write_to_res=('results.txt', get_name(config)))

        # rescore
        with wb.processing('rescore...'):
            data.write_data(nbest.get_nbest_list(data), os.path.join(workdir, 'nbest.txt'))
            nbest.lmscore = m.rescore(nbest.get_nbest_list(data))
            # print(len(nbest.lmscore))
            nbest.write_lmscore(os.path.join(workdir, 'rescore.lmscore'))

        # tune lm-scale
        with wb.processing('wer...'):
            wer = nbest.wer()
            nbest.write_log(os.path.join(workdir, 'rescore.log'))
            nbest.write_1best(os.path.join(workdir, 'rescore.best'))

            print('wer={} lmscale={}'.format(wer, nbest.lmscale))
            fres = wb.FRes('results.txt')
            fres.AddWER(get_name(config), wer)
            fres.Add(get_name(config), ['lmscale'], [nbest.lmscale])

if __name__ == '__main__':
    main()
