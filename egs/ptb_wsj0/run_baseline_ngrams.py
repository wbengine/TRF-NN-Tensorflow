import os
import sys
import numpy as np

from base import *
from lm import ngramlm

res_file = 'results.txt'
fres = wb.FRes(res_file)  # the result file


def main():

    data = reader.Data().load_raw_data(reader.ptb_raw_dir(),
                                       add_beg_token='<s>',
                                       add_end_token='</s>')
    nbest = reader.NBest(*reader.wsj0_nbest())

    config = ngramlm.Config(data)
    config.res_file = 'results.txt'

    if wb.is_window():
        bindir = 'd:\\wangbin\\tools'
    else:
        bindir = '../../tools/srilm'

    order_reg = [3, 4, 5]
    for order in order_reg:
        config.order = order
        workdir = 'ngramlm/' + str(config)
        m = ngramlm.Model(config, data, bindir, workdir, name=str(config))

        print('train...')
        m.train(write_to_res=(res_file, str(config)))

        print('rescore...')
        nbest.lmscore = m.rescore(nbest.get_nbest_list(data))
        nbest.write_lmscore(os.path.join(workdir, 'nbest.lmscore'))
        wer = nbest.wer()
        print('wer={} lmscale={}, acscale={}'.format(wer, nbest.lmscale, nbest.acscale))
        fres.AddWER(str(config), wer)

if __name__ == '__main__':
    main()
