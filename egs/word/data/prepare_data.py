import os
import random
import numpy as np

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


def main():
    src_file = 'LDC_gigaword_en.words'  # this file is exact from LDC gigaword english corpus
    write_train = 'train.words'
    write_valid = 'valid.words'
    write_test = 'test.words'

    write_nbest = 'nbest.words'
    write_temp = 'transcript.words'

    np.random.seed(0)
    m2 = 2000  # valid size
    m3 = 2000  # test size
    with open(src_file, 'rt') as f, \
        open(write_train, 'wt') as f1, \
        open(write_valid, 'wt') as f2, \
        open(write_test, 'wt') as f3:
        a = f.read().split('\n')
        for i in range(m2):
            t = np.random.randint(len(a))
            write_line(f2, a[t])
            del a[t]
        for i in range(m3):
            t = np.random.randint(len(a))
            write_line(f3, a[t])
            del a[t]
        for s in a:
            write_line(f1, s)
    print('all  ={}'.format(count_line(src_file)))
    print('train={}'.format(count_line(write_train)))
    print('valid={}'.format(count_line(write_valid)))
    print('test ={}'.format(count_line(write_test)))

    # generate nbest list
    n = 20 # n-best list
    label = 'EDWORD'
    vlist = get_vocab([write_train, write_valid, write_test])
    print('v size = ', len(vlist))
    print(vlist)
    with open(write_test, 'rt') as f, open(write_nbest, 'wt') as f1, open(write_temp, 'wt') as f2:
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









