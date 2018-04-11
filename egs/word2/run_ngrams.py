import os
import sys
import numpy as np
import json

from base import *
from lm import ngramlm


res_file = 'results.txt'
fres = wb.FRes(res_file)  # the result file

train = 'data/train.chr'
valid = 'data/valid.chr'
test = valid
nbest_files = ('data/nbest.words', 'data/transcript.words')


def main():

    data = reader.Data().load_raw_data([train, valid, test],
                                       add_beg_token='<s>',
                                       add_end_token='</s>')
    nbest = reader.NBest(*nbest_files)

    config = ngramlm.Config(data)
    config.res_file = 'results.txt'
    config.discount = '-wbdiscount'

    if wb.is_window():
        bindir = 'd:\\wangbin\\tools'
    else:
        bindir = '../../tools/srilm'
    workdir = 'ngramlm/' + str(config)

    order_reg = [3, 4, 5, 6]
    for order in order_reg:
        config.order = order
        m = ngramlm.Model(config, data, bindir, workdir, name=str(config))

        print('train...')
        m.train(write_to_res=(res_file, str(config)))

        print('rescore...')
        nbest.lmscore = m.rescore(nbest.get_nbest_list(data))
        wer = nbest.wer()
        print('wer={} lmscale={}, acscale={}'.format(wer, nbest.lmscale, nbest.acscale))
        fres.AddWER(str(config), wer)

if __name__ == '__main__':
    main()
