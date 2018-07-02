import os

from base import *

# trf_log = {1: 'trf_t1_nce/trf_nce_noise4_data0.5_e200_blstm_200x1_pred_logzlinear_LSTMLenGen',
#            5: 'trf_t5_nce/trf_nce_noise4_data0.1_e200_blstm_200x1_pred_logzlinear_LSTMLenGen',
#            10: 'trf_t10_nce/trf_nce_noise4_data0.5_e200_blstm_200x1_pred_logzlinear_LSTMLenGen',
#            100: 'trf_t100_nce/trf_nce_noise4_data0.5_e200_blstm_200x1_pred_logzlinear_LSTMLenGen'}
#
# for key, s in trf_log.items():
#     wer_log = os.path.join(s, 'wer_per_epoch.log')
#     fres = wb.FRes(wer_log)
#     fres.Read()
#     wers = [float(a[1]) for a in fres.data]
#     opt_wer = min(wers)
#     print('t{}, opt_wer={}'.format(key, opt_wer))

lmscores = {'trf': 'trf/epoch5.00.test.lmscore',
            'kn5': 'ngramlm/t100_KN5_00225/nbest.lmscore',
            'lstm': 'lstmlm/t100_lstm_e256_h1024x2_AdaptiveSoftmax_adam/lmscores/epoch2.00.test.lmscore',
            }

kn5_lm = os.path.join(os.path.split(lmscores['kn5'])[0], 't100_KN5_00225.lm')
with open(kn5_lm, 'rt') as f:
    flag = False
    ngrams = [0] * 10
    for line in f:
        if len(line.split()) == 0:
            continue

        if line.split()[0] == '\\data\\':
            flag = True
            continue
        if flag:
            s = line.split()
            if s[0] == 'ngram':
                a = line.split()[1].split('=')
                order = int(a[0])
                num = int(a[1])
                ngrams[order] = num
            else:
                break

print(ngrams)
print('kn5 param = ', np.sum(ngrams))


nbest_test = reader.NBest(*reader.wsj0_nbest())


combine_type = ['kn5', 'lstm', 'trf', 'lstm+kn5','trf+kn5', 'trf+lstm', 'trf+lstm+kn5']
for type_str in combine_type:
    name_list = type_str.split('+')
    n = len(name_list)
    score_list = [wb.LoadScore(lmscores[name]) for name in name_list]
    s = np.array(score_list).sum(axis=0)
    nbest_test.lmscore = s
    wer = nbest_test.wer()
    print('name={} wer={:.2f} lmscale={:.2f}'.format(type_str, wer, nbest_test.lmscale))