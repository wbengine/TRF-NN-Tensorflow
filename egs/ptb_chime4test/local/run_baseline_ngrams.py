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


def main():
    nbest_cmp = task.NBestComputer()
    data = reader.Data().load_raw_data([task.train, task.valid, task.test],
                                       add_beg_token='<s>',
                                       add_end_token='</s>')

    config = ngramlm.Config(data)
    config.res_file = 'results.txt'

    order_reg = [2, 3]
    for order in order_reg:
        config.order = order
        config.cutoff = [0] * order

        workdir = wb.mkdir('ngramlm/' + str(config), is_recreate=False)
        sys.stdout = wb.std_log(workdir + '/ngram.log')
        print(workdir)

        m = ngramlm.Model(config, data, bindir, workdir)

        # train
        print('training...')
        m.train()

        # rescore
        print('rescoring...')
        time_beg = time.time()
        for nbest in nbest_cmp.nbests:
            nbest.lmscore = m.rescore(nbest.get_nbest_list(data))
            # print(len(nbest.lmscore))
        nbest_cmp.write_lmscore(workdir + '/model')
        print('rescore time={:.2f}m'.format((time.time() - time_beg)/60))

        # tune lm-scale
        print('computing wer...')
        nbest_cmp.cmp_wer()
        nbest_cmp.write_to_res(config.res_file, str(config))

if __name__ == '__main__':
    main()
