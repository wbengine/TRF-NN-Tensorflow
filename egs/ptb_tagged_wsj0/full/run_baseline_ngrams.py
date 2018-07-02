import os
import sys
import numpy as np
import json

from base import *
from lm import ngramlm


res_file = 'results.txt'
fres = wb.FRes(res_file)  # the result file

with open('data.info') as f:
    data_info = json.load(f)


def main():

    data = seq.Data(vocab_files=data_info['vocab'],
                    train_list=data_info['train'],
                    valid_list=data_info['valid'],
                    test_list=data_info['test']
                    )

    data = data.create_data()
    nbest = reader.NBest(*data_info['nbest'])

    config = ngramlm.Config(data)
    config.res_file = res_file

    if wb.is_window():
        bindir = 'd:\\wangbin\\tools'
    else:
        bindir = '../../../tools/srilm'
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
