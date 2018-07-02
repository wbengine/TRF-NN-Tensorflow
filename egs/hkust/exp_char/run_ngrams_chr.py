import os
import sys
import json
import numpy as np
import time
from eval import nbest_eval_chr

from base import *
from lm import *

if wb.is_window():
    bindir = 'd:\\wangbin\\tools'
else:
    bindir = '../../../tools/srilm/'


with open('data.info') as f:
    info = json.load(f)


def get_name(config):
    return str(config)+'_chr'


def main():
    data = reader.Data().load_raw_data(file_list=info['hkust_train_chr'] +
                                                 info['hkust_valid_chr'] +
                                                 info['hkust_valid_chr'],
                                       add_beg_token='<s>',
                                       add_end_token='</s>'
                                       )
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

        # wer
        nbest_eval_chr(m, data, workdir,
                       res_file='results.txt',
                       res_name=get_name(config))

if __name__ == '__main__':
    main()
