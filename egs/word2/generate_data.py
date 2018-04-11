import os
import random
import numpy as np

from base import *


def write_line(f, line):
    a = line.split()
    if len(a) > 0:
        s = a[0] # remove '\n'
        f.write(' '.join(list(s)) + '\n')

    
def count_line(file):
    with open(file) as f:
        return len(f.read().split('\n'))


def get_vocab(file_list):
    v = dict()
    for fname in file_list:
        f = open(fname)
        for line in f:
            for w in line.split():
                v[w] = 0
        f.close()
    return sorted(v.keys())


def random_line(seq, vlist, n):
    res = []
    res.append(seq)
    while len(res) < n:
        s = list(seq)
        err_type = np.random.randint(3)
        pos = np.random.randint(len(s))
        if err_type == 0 and len(s) < 26:
            # insert
            s.insert(pos, np.random.choice(vlist))
        elif err_type == 1 and len(s) > 1:
            # delete
            del s[pos]
        else:
            # substitute
            s[pos] = np.random.choice(vlist)
        res.append(s)
    np.random.shuffle(res)
    return res


def read_data(file):
    chr_list = []
    with open(file, 'rt') as f:
        for line in f:
            a = list(line.split()[0])
            chr_list.append(a)

    return chr_list


def write_to_chr_tag(chr_list, file):
    with open(file + '.chr', 'wt') as fchr, open(file+'.tag', 'wt') as ftag:
        for cs in chr_list:
            fchr.write(' '.join(cs) + '\n')

            tag = []
            for c in cs:
                if c >= 'a' and c <= 'g':
                    tag.append('t1')
                elif c >= 'h' and c <= 'n':
                    tag.append('t2')
                else:
                    tag.append('t3')
            ftag.write(' '.join(tag) + '\n')


def main():
    src_file = '../word/data/LDC_gigaword_en.words'  # this file is exact from LDC gigaword english corpus
    write_train = 'data/train'
    write_valid = 'data/valid'
    write_test = 'data/test'

    train_size = 10000
    valid_size = 200
    test_size = 200

    write_nbest = 'data/nbest.words'
    write_temp = 'data/transcript.words'

    wb.mkdir('data')

    chr_list = read_data(src_file)
    idx = list(range(len(chr_list)))

    np.random.seed(0)
    np.random.shuffle(idx)

    valid_idx = idx[0: valid_size]
    test_idx = idx[valid_size: valid_size + test_size]
    train_idx = idx[valid_size + test_size:][0: train_size]

    write_to_chr_tag([chr_list[i] for i in train_idx] + ['a'], write_train)
    write_to_chr_tag([chr_list[i] for i in valid_idx], write_valid)
    write_to_chr_tag([chr_list[i] for i in test_idx], write_test)

    # generate nbest list
    n = 20  # n-best list
    label = 'EDWORD'
    vlist = get_vocab([write_train + '.chr'])
    print('v size = ', len(vlist))
    print(vlist)
    with open(write_test + '.chr', 'rt') as f, open(write_nbest, 'wt') as f1, open(write_temp, 'wt') as f2:
        nline = 0
        for line in f:
            seq = line.split()
            nbest = random_line(seq, vlist, n)
            for i in range(len(nbest)):
                f1.write('{}{}-{}\t'.format(label, nline, i+1) + ' '.join(nbest[i]) + '\n')
            f2.write('{}{}\t'.format(label, nline) + ' '.join(seq) + '\n')
            nline += 1

if __name__ == '__main__':
    main()









