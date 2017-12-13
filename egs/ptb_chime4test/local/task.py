import numpy as np
import os
import time

from base import *


train = '../../ptb_wsj0/data/ptb/ptb.train.txt'
valid = '../../ptb_wsj0/data/ptb/ptb.valid.txt'
test = '../../ptb_wsj0/data/ptb/ptb.test.txt'

# train_sents = 1625584
# train_words = 37243300
# train_vocab = 4983
# valid_sents = 3280
# valid_words = 54239
# valid_vocab = 1596

nbest_dir = '../../CHiME4/data/nbest/nbest_mvdr_stack_heq_multi/'
nbest_files = ['words_text',   # nbest list
               'text',         # correct
               'acwt',         # acoustic score
               'lmwt.lmonly',  # lm score
               'lmwt.nolm'     # graph score
               ]


# merge the nbest list
def merge_files(input_files, output_file):
    with open(output_file, 'wt') as fout:
        for fname in input_files:
            print('merge file: {} -> {}'.format(fname, output_file))
            f = open(fname, 'rt')
            for line in f:
                fout.write(line)
            f.close()

# merge the nbests
# new_nbest_dir = 'pre/'
# wb.mkdir(new_nbest_dir)
# dt_to_dir = new_nbest_dir + 'dt05_simu_real/'
# et_to_dir = new_nbest_dir + 'et05_simu_real/'
# if not wb.exists(dt_to_dir):
#     wb.mkdir(dt_to_dir)
#     wb.mkdir(et_to_dir)
#     for file_name in nbest_files:
#         merge_files([nbest_dir + 'nbestlist_dt05_simu/' + file_name,
#                      nbest_dir + 'nbestlist_dt05_real/' + file_name],
#                     dt_to_dir + file_name)
#         merge_files([nbest_dir + 'nbestlist_et05_simu/' + file_name,
#                      nbest_dir + 'nbestlist_et05_real/' + file_name],
#                     et_to_dir + file_name)
#
#
# nbest_dt = [dt_to_dir + s for s in nbest_files]
# nbest_et = [et_to_dir + s for s in nbest_files]

nbest_all = [None] * 4
nbest_all[0] = [nbest_dir + 'nbestlist_dt05_real/' + s for s in nbest_files]
nbest_all[1] = [nbest_dir + 'nbestlist_dt05_simu/' + s for s in nbest_files]
nbest_all[2] = [nbest_dir + 'nbestlist_et05_real/' + s for s in nbest_files]
nbest_all[3] = [nbest_dir + 'nbestlist_et05_simu/' + s for s in nbest_files]


class NBestComputer(object):
    def __init__(self, lmscale=list(range(1, 20))):
        self.lmscale_vec = lmscale

        self.nbests = []
        for i in range(len(nbest_all)):
            self.nbests.append(reader.NBest(*nbest_all[i]))

        self.lmscale = 1.0
        self.wer = np.zeros(len(nbest_all))

        # get info
        for nbest in self.nbests:
            info = wb.TxtInfo(nbest.nbest)
            info.nWord -= info.nLine
            info.min_len -= 1
            info.max_len -= 1
            info.nVocab -= info.nLine
            print('  ' + nbest.nbest + ': ' + str(info))

    def cmp_dev_wer(self):
        # tune the lmscale
        opt_lmscale = 0
        opt_wer = 100
        for w in self.lmscale_vec:
            self.nbests[0].wer(lmscale=[w], rm_unk=True)
            self.nbests[1].wer(lmscale=[w], rm_unk=True)

            wer = self.get_valid_wer()
            if wer < opt_wer:
                opt_wer = wer
                opt_lmscale = w

        self.lmscale = opt_lmscale
        return opt_wer

    def cmp_wer(self):
        # tune the lmscale
        self.cmp_dev_wer()

        # compute the WER for all nbests
        for i in range(len(nbest_all)):
            self.wer[i] = self.nbests[i].wer(lmscale=[self.lmscale], rm_unk=True)

        return self.get_test_wer()

    def get_valid_wer(self):
        test_wer = (self.nbests[0].total_err + self.nbests[1].total_err) / \
                   (self.nbests[0].total_word + self.nbests[1].total_word)
        return 100.0 * test_wer

    def get_test_wer(self):
        test_wer = (self.nbests[2].total_err + self.nbests[3].total_err) / \
                   (self.nbests[2].total_word + self.nbests[3].total_word)
        return 100.0 * test_wer

    def write_to_res(self, res_file, res_name):
        res = wb.FRes(res_file)
        res.Add(res_name, ['lm-scale'], [self.lmscale])
        res.Add(res_name, ['dt', 'et'], [self.get_valid_wer(), self.get_test_wer()])
        res.Add(res_name, ['dt_real', 'dt_simu', 'et_real', 'et_simu'], self.wer.tolist())

    def write_lmscore(self, fname='score'):
        for nbest, name in zip(self.nbests, ['dt_real', 'dt_simu', 'et_real', 'et_simu']):
            nbest.write_lmscore(fname + '.' + name+'.lmscore')


class Ops(wb.Operation):
    def __init__(self, m, wer_per_epoch=1.0):
        self.m = m
        self.nbest_cmp = NBestComputer()
        self.wer_next_epoch = 0
        self.wer_per_epoch = wer_per_epoch
        self.opt_det_wer = 100
        self.opt_txt_wer = 100
        self.write_models = wb.mkdir(os.path.join(self.m.logdir, 'wer_results'))

    def run(self, step, epoch):
        super().run(step, epoch)

        if epoch >= self.wer_next_epoch:
            self.wer_next_epoch = (int(epoch / self.wer_per_epoch) + 1) * self.wer_per_epoch
            epoch_num = int(int(epoch / self.wer_per_epoch) * self.wer_per_epoch)

            print('[Ops] rescoring:', end=' ', flush=True)

            # resocring
            with self.m.time_recoder.recode('rescore'):
                time_beg = time.time()
                for nbest in self.nbest_cmp.nbests:
                    nbest.lmscore = -self.m.get_log_probs(nbest.get_nbest_list(self.m.data))
                rescore_time = time.time() - time_beg
            # compute wer
            with self.m.time_recoder.recode('wer'):
                time_beg = time.time()
                self.nbest_cmp.cmp_wer()
                self.nbest_cmp.write_to_res(os.path.join(self.m.logdir, 'wer_per_epoch.log'), 'epoch%d' % epoch_num)
                dev_wer = self.nbest_cmp.get_valid_wer()
                tst_wer = self.nbest_cmp.get_test_wer()
                wer_time = time.time() - time_beg
                print('epoch={:.2f} dev_wer={:.2f} test_wer={:.2f} lmscale={} '
                      'rescore_time={:.2f}, wer_time={:.2f}'.format(
                    epoch, dev_wer, tst_wer, self.nbest_cmp.lmscale,
                    rescore_time / 60, wer_time / 60))

            # write models
            if dev_wer < self.opt_det_wer:
                self.opt_det_wer = dev_wer

            self.m.save(self.write_models + '/epoch%d' % epoch_num)
            self.nbest_cmp.write_lmscore(self.write_models + '/epoch%d' % epoch_num)
