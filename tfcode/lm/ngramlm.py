################################################
# This is a wrapper suing the executable files of SRILM toolkits
###############################################

import os
import sys
import numpy as np

from base import *


# exact the vocabulary form the corpus
def GetVocab(fname, v, unk='<UNK>'):
    f = open(fname, 'rt')
    for line in f:
        a = line.upper().split()
        for w in a:
            v.setdefault(w, 0)
            v[w] += 1
    if unk not in v:
        v[unk] = 0
    f.close()
    return v


# set the vocab id
def SetVocab(v):
    n = 2
    for k in sorted(v.keys()):
        v[k] = n
        n += 1
    return v


# trans txt corpus to id corpus
def CorpusToID(fread, fwrite, v, unk='<UNK>'):
    print('[w2id] ' + fread + ' -> ' + fwrite)
    f1 = open(fread, 'rt')
    f2 = open(fwrite, 'wt')
    for line in f1:
        a = line.upper().split()
        for w in a:
            if w in v:
                f2.write('{} '.format(v[w]))
            else:
                f2.write('{} '.format(v[unk]))
        f2.write('\n')
    f1.close()
    f2.close()


# trans id to txt
def CorpusToW(fread, fwrite, v):
    print('[id2w] ' + fread + ' -> ' + fwrite)
    v1 = [''] * (len(v) + 2)
    for key in v.keys():
        v1[v[key]] = key

    f1 = open(fread, 'rt')
    f2 = open(fwrite, 'wt')
    for line in f1:
        a = line.split()
        for w in a:
            f2.write('{} '.format(v1[int(w)]))
        f2.write('\n')
    f1.close()
    f2.close()


# write vocabulary
def WriteVocab(fname, v):
    f = open(fname, 'wt')
    vlist = sorted(v.items(), key=lambda d: d[1])
    f.write('<s>\n</s>\n')
    for w, wid in vlist:
        f.write('{}\t{}\n'.format(wid, w))
    f.close()


# read vocabulary
def ReadVocab(fname):
    v = dict()
    f = open(fname, 'rt')
    f.readline()
    f.readline()
    for line in f:
        a = line.split()
        v[a[1].upper()] = int(a[0])
    f.close()
    return v


# trans nbest list to id files
def GetNbest(ifile, ofile, v, unk='<UNK>'):
    print('[nbest] ' + ifile + ' -> ' + ofile)
    fin = open(ifile, 'rt')
    fout = open(ofile, 'wt')
    for line in fin:
        a = line.upper().split()
        for w in a[1:]:
            nid = 0
            if w in v:
                nid = v[w]
            elif unk in v:
                nid = v[unk]
            else:
                print('[error] on word in vocabulary ' + w)
            fout.write('{} '.format(nid))
        fout.write('\n')
    fin.close()
    fout.close()


# trans the debug 2 output of SRILM to sentence score
def Debug2SentScore(fdbg, fscore):
    with open(fdbg, 'rt') as f1, open(fscore, 'wt') as f2:
        score = []
        for line in f1:
            if 'logprob=' not in line:
                continue
            s = -float(line[line.find('logprob='):].split()[1])
            # log10 to log
            score.append(s / np.log10(np.exp(1)))
        for i in range(len(score)-1):
            f2.write('sent={}\t{}\n'.format(i, score[i]))


class Config:
    def __init__(self, data):
        self.order = 3
        self.discount = '-kndiscount'
        self.cutoff = [0, 0, 0]
        self.res_file = None  # the result files used to write the results

    def __str__(self):
        return 'KN{}_{}'.format(self.order, ''.join([str(i) for i in self.cutoff]))


class Model:
    def __init__(self, config, data, bindir, workdir, name='ngram'):
        self.config = config
        self.data = data
        self.name = name
        self.workdir = os.path.join(workdir, '')
        self.bindir = os.path.join(bindir, '')
        wb.mkdir(workdir)

        # write id to files
        name_list = ['train', 'valid', 'test']
        for seq_list, fname in zip(self.data.datas, name_list):
            self.data.write_data(seq_list, self.workdir + fname + '.id',
                                 skip_beg_token=True,
                                 skip_end_token=True)
        # write vocab to files
        with open(self.workdir + 'vocab', 'wt') as f:
            f.write('<s>\n</s>\n')
            for i in range(2, self.data.get_vocab_size()):
                f.write('{}\t{}\n'.format(i, self.data.word_list[i]))

    def train(self):
        write_count = self.workdir + self.name + '.count'
        write_model = self.workdir + self.name + '.lm'

        if wb.exists(write_model):
            print('model exist, skip training')
            return

        cutoff_cmd = ' '.join(['-gt{}min {}'.format(i+1, n) for i, n in enumerate(self.config.cutoff)])

        cmd = self.bindir + 'ngram-count '
        cmd += ' -text {0}train.id -vocab {0}vocab'.format(self.workdir)
        cmd += ' -order {} -write {} '.format(self.config.order, write_count)
        cmd += cutoff_cmd + ' '
        os.system(cmd)

        cmd = self.bindir + 'ngram-count '
        cmd += ' -vocab {}vocab'.format(self.workdir)
        cmd += ' -read {}'.format(write_count)
        cmd += ' -order {} -lm {} '.format(self.config.order, write_model)
        cmd += self.config.discount + ' -interpolate ' + cutoff_cmd
        os.system(cmd)

        # get ppl
        if self.config.res_file is not None:
            PPL = [0] * 3
            testno = [self.workdir + s + '.id' for s in ['train', 'valid', 'test']]
            for i in range(min(len(self.data.datas), len(testno))):
                PPL[i] = self.ppl(testno[i], type='id')
            res_file = wb.FRes(self.config.res_file)
            res_file.AddPPL(str(self.config), PPL, testno)

    def ppl(self, fname, type='txt'):
        """
        calculate the PPL of a file
        Args:
            fname: str, file name
            type:  str, type of file, 'txt' or 'id'

        Returns:
            the ppl of input files
        """
        if type == 'txt':
            # get the id files
            _, file_name = os.path.split(fname)
            id_file = self.workdir + file_name + '.ppl.id'

            seq_list = self.data.load_data(fname)
            self.data.write_data(seq_list, id_file,
                                 skip_beg_token=True,
                                 skip_end_token=True)
        else:
            id_file = fname

        cmd = self.bindir + 'ngram -order {} -lm {} -ppl {}'.format(self.config.order,
                                                                    self.workdir + self.name + '.lm',
                                                                    id_file)
        res = os.popen(cmd).read()
        print(res)
        return float(res[res.find('ppl='):].split()[1])
    
    def rescore(self, nbest_list, temp_name='nbest'):
        """
        rescore the nbest
        Args:
            nbest_list: id list
            temp_name:

        Returns:

        """
        write_nbest_id = self.workdir + temp_name + '.id'
        write_temp = self.workdir + temp_name + '.debug'
        write_lmscore = self.workdir + temp_name + '.lmscore'

        # if the nbest_list contain empty line
        # pad an unk
        for i in range(len(nbest_list)):
            if len(nbest_list[i]) == 2:  # onle beg/end tokens
                nbest_list[i].insert(1, self.data.get_unk_token())
        self.data.write_data(nbest_list, write_nbest_id, skip_beg_token=True, skip_end_token=True)
        
        cmd = self.bindir + 'ngram -lm {} -order {} -ppl {} -debug 2 > {}'.format(self.workdir + self.name + '.lm',
                                                                                  self.config.order,
                                                                                  write_nbest_id, write_temp)
        os.system(cmd)
        Debug2SentScore(write_temp, write_lmscore)

        return wb.LoadScore(write_lmscore)
        
        
        
        
        
        
