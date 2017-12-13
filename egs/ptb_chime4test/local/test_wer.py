import tensorflow as tf
import os
import sys
import time
import numpy as np

import task
from base import *
from lm import *

workdirs = ['.',
            '/mnt/workspace2/wangbin/server12_work/TRF-NN-tensorflow/egs/ptb_chime4test/local',
            '/mnt/workspace/wangbin/server9_work/TRF-NN-tensorflow/egs/ptb_chime4test/local']


def search_file(name):
    for workdir in workdirs:
        s = os.path.join(workdir, name)
        if wb.exists(s):
            print('load %s' % s)
            return s
    raise TypeError('Can not find file: %s' % name)


ngram_dir = 'ngramlm/KN5_00000'
lstm_dir = 'lstm/lstm_e200_h200x2'

# trf_dir = 'trf_nce_new/trf_nce10_e200_blstm_200x2_pred_noise2gram'
trf_dir = 'trf_nce_new/trf_nce10_featg2_e128_cnn_(1to10)x128_(3x128)x3_relu_noise2gram'
trf_epoch = 80
res_file = wb.mkdir('res') + '/LSTM2x200_KN5_TRF_cnn.wers'


sys_labels = ['dt_real', 'dt_simu', 'et_real', 'et_simu']
ngram_score_files = [ngram_dir + '/model.{}.lmscore'.format(t) for t in sys_labels]
lstm_score_files = [lstm_dir + '/model.{}.lmscore'.format(t) for t in sys_labels]
trf_score_files = [search_file(trf_dir + '/wer_results/epoch{}.{}.lmscore'.format(trf_epoch, t)) for t in sys_labels]


res = wb.FRes(res_file)
res.AddComment('ngram = ' + ngram_dir)
res.AddComment('lstm  = ' + lstm_dir)
res.AddComment('qTRF  = ' + trf_dir)

ngram_name = 'KN5'
lstm_name = 'LSTM'
trf_name = 'qTRF'


nbest_cmp = task.NBestComputer()


def set_score_cmp_wer(fun_score, res_name):
    print(res_name)
    for i, nbest in enumerate(nbest_cmp.nbests):
        nbest.lmscore = fun_score(i)
    nbest_cmp.cmp_wer()
    print(' ', sys_labels)
    print(' ', nbest_cmp.wer.tolist())
    print(' ', 'valid=', nbest_cmp.get_valid_wer())
    print(' ', 'test=', nbest_cmp.get_test_wer())
    nbest_cmp.write_to_res(res_file, res_name)


# single models
set_score_cmp_wer(lambda i: wb.LoadScore(ngram_score_files[i]), ngram_name)
set_score_cmp_wer(lambda i: wb.LoadScore(lstm_score_files[i]), lstm_name)
set_score_cmp_wer(lambda i: wb.LoadScore(trf_score_files[i]), trf_name)

# KN + lstm
set_score_cmp_wer(lambda i: 1/2 * wb.LoadScore(ngram_score_files[i]) + 1/2 * wb.LoadScore(lstm_score_files[i]),
                  ngram_name + '+' + lstm_name)

# KN + trf
set_score_cmp_wer(lambda i: 1/2 * wb.LoadScore(ngram_score_files[i]) + 1/2 * wb.LoadScore(trf_score_files[i]),
                  ngram_name + '+' + trf_name)

# KN + lstm + trf
set_score_cmp_wer(lambda i: 1/3 * wb.LoadScore(ngram_score_files[i]) +
                            1/3 * wb.LoadScore(lstm_score_files[i]) +
                            1/3 * wb.LoadScore(trf_score_files[i]),
                  ngram_name + '+' + lstm_name + '+' + trf_name)

# lstm + trf
set_score_cmp_wer(lambda i: 1/2 * wb.LoadScore(lstm_score_files[i]) + 1/2 * wb.LoadScore(trf_score_files[i]),
                  lstm_name + '+' + trf_name)



