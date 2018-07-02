from base import *
from eval import nbest_eval_lmscore_chr

res_file = 'results_inter.txt'

lstm_path = ['lstm/lstm_e200_h200x2_chr/nbest.%s.lmscore' % flag for flag in ['valid', 'test']]
kn5_path = ['ngramlm/KN5_00000_chr/nbest.%s.lmscore' % flag for flag in ['valid', 'test']]

trf_root = 'trf_nce/trf_nce_noise4_data1_e200_blstm_200x1_pred_logzlinear_LSTMLenGen/lmscores'
# trf_epoch = 40
trf_avg_epoch = [44, 45, 46]

trf_lmscore = None
for n in trf_avg_epoch:
    trf_path = [trf_root + '/epoch{:.2f}.{}.lmscore'.format(n, s) for s in ['valid', 'test']]
    if trf_lmscore is None:
        trf_lmscore = [wb.LoadScore(s) for s in trf_path]
    else:
        trf_lmscore[0] += wb.LoadScore(trf_path[0])
        trf_lmscore[1] += wb.LoadScore(trf_path[1])

for s in trf_lmscore:
    s /= len(trf_avg_epoch)


lstm_kn5 = [wb.ScoreInterpolate(s1, s2, 0.5) for s1, s2 in zip(lstm_path, kn5_path)]
lstm_trf = [wb.ScoreInterpolate(s1, s2, 0.5) for s1, s2 in zip(lstm_path, trf_lmscore)]
kn5_trf = [wb.ScoreInterpolate(s1, s2, 0.5) for s1, s2 in zip(kn5_path, trf_lmscore)]
lstm_kn5_trf = [(wb.LoadScore(s1) + wb.LoadScore(s2) + s3)/3 for s1, s2, s3 in zip(lstm_path, kn5_path, trf_lmscore)]

# nbest_eval_lmscore_chr(kn5_path, res_file, 'kn5')
# nbest_eval_lmscore_chr(lstm_path, res_file, 'lstm')
nbest_eval_lmscore_chr(trf_lmscore, res_file, 'trf')
# nbest_eval_lmscore_chr(lstm_kn5, res_file, 'lstm+kn5')
nbest_eval_lmscore_chr(lstm_trf, res_file, 'lstm+trf')
nbest_eval_lmscore_chr(kn5_trf, res_file, 'kn5+trf')
nbest_eval_lmscore_chr(lstm_kn5_trf, res_file, 'lstm+kn5+trf')