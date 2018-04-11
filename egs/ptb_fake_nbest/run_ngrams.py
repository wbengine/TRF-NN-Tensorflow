import os
import sys
import numpy as np

from base import *
from lm import ngramlm
from gen_data import train_files, nbest_eval

res_file = 'results.txt'
fres = wb.FRes(res_file)  # the result file


def main():

    data = reader.Data().load_raw_data(train_files,
                                       add_beg_token='<s>',
                                       add_end_token='</s>')
    # nbest_real = reader.NBest(*reader.wsj0_nbest())
    # nbest_fake = reader.NBest(*nbest_files)

    config = ngramlm.Config(data)
    config.res_file = 'results.txt'

    if wb.is_window():
        bindir = 'd:\\wangbin\\tools'
    else:
        bindir = '../../tools/srilm'

    order_reg = [5, 6]
    for order in order_reg:
        config.order = order
        workdir = 'ngramlm/' + str(config)
        m = ngramlm.Model(config, data, bindir, workdir, name=str(config))

        print('train...')
        m.train(write_to_res=(res_file, str(config)))

        print('rescore...')
        nbest_eval(m, data, workdir, fres, str(config))

if __name__ == '__main__':
    main()
