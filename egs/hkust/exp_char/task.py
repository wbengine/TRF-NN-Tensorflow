import os
import sys
import shutil
import json
import re
import numpy as np

from base import *


def process_pku_data(input, output, max_line=None):
    with open(output, 'wt') as fout, open(input, 'rt') as fsrc:
        read_line = 0
        for line in fsrc:
            wfull_list = line.split()
            chr_list = []
            pos_list = []
            for wfull in wfull_list:
                # remove the pronounce {wangbin}
                if wfull[-1] == '}':
                    i = wfull.rfind('{')
                    if i == -1:
                        w = re.search('[^a-zA-Z]+', wfull).group()
                        print('error in word={}, w={}'.format(wfull, w))
                    else:
                        w = wfull[0: i]
                else:
                    w = wfull

                chrs = wb.split_to_char_ch(w, keep_en_words=False)
                poss = wb.generate_pos(len(chrs))
                chr_list += chrs
                pos_list += poss

                if len(chr_list) != len(pos_list):
                    raise TypeError('split error! wfull={} w={} chrs={} poss={}'.format(wfull, w, chrs, poss))

                flag = False
                if w in ['，', '。', '？', '！', '；']:
                    flag = True

                if flag:
                    # write chrs and poss
                    fout.write(' '.join(['{:<1}'.format(c) for c in chr_list]) + '\n')
                    fout.write(' '.join(['{:<2}'.format(c) for c in pos_list]) + '\n')
                    chr_list = []
                    pos_list = []

            if chr_list and pos_list:
                # write chrs and poss
                fout.write(' '.join(['{:<1}'.format(c) for c in chr_list]) + '\n')
                fout.write(' '.join(['{:<2}'.format(c) for c in pos_list]) + '\n')

            read_line += 1
            if max_line is not None and read_line >= max_line:
                break


def process_hkust_data(input, output, add_pos=True):
    with open(input, 'rt') as fin, open(output, 'wt') as fout:
        for line in fin:
            w_list = line.split()
            chr_list = []
            pos_list = []

            for w in w_list:
                if w[0] == '[' and w[-1] == ']':
                    continue
                chrs = wb.split_to_char_ch(w)
                poss = wb.generate_pos(len(chrs))
                chr_list += chrs
                pos_list += poss

            if chr_list and pos_list:
                fout.write(' '.join(['{:<1}'.format(c) for c in chr_list]) + '\n')
                if add_pos:
                    fout.write(' '.join(['{:<2}'.format(c) for c in pos_list]) + '\n')


def split_to_train_valid_test(all_data, train_file, valid_file, test_file, valid_rate=0.01, test_rate=0.01):
    data = seq.DataX(total_level=2, train_list=[all_data], beg_token='<s>', end_token='</s>', unk_token='<unk>')
    data.vocabs[0].write(os.path.join(root_dir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(root_dir, 'vocab.pos'))

    n = len(data.datas[0])
    n_valid = int(n * valid_rate)
    n_test = int(n * test_rate)
    n_train = n - n_valid - n_test

    a = np.arange(n)
    np.random.shuffle(a)

    train_seqs = [data.datas[0][i] for i in a[0: n_train]]
    valid_seqs = [data.datas[0][i] for i in a[n_train: n_train + n_valid]]
    test_seqs = [data.datas[0][i] for i in a[n_train + n_valid:]]
    write_text(train_seqs, data.vocabs, train_file)
    write_text(valid_seqs, data.vocabs, valid_file)
    write_text(test_seqs, data.vocabs, test_file)


def write_text(seq_list, vocabs, file_name):
    with open(file_name, 'wt') as fout:
        for seq in seq_list:
            chr_list = vocabs[0].ids_to_words(seq.x[0][1:-1])
            pos_list = vocabs[1].ids_to_words(seq.x[1][1:-1])
            fout.write(' '.join(['{:<1}'.format(c) for c in chr_list]) + '\n')
            fout.write(' '.join(['{:<2}'.format(c) for c in pos_list]) + '\n')


def nbest_to_char(nbest_files, output_files):

    def copy_nbest(input, output):
        with open(input) as fin, open(output, 'wt') as fout:
            for line in fin:
                a = line.split()
                label = a[0]
                w_list = a[1:]
                chr_list = []

                for w in w_list:
                    if w[0] == '[' and w[-1] == ']':
                        continue
                    chr_list += wb.split_to_char_ch(w)

                fout.write(label + ' ' + ' '.join(chr_list) + '\n')

    copy_nbest(nbest_files[0], output_files[0])
    copy_nbest(nbest_files[1], output_files[1])
    for f1, f2 in zip(nbest_files[2:], output_files[2:]):
        shutil.copy(f1, f2)


def split_nbest_to_valid_test(input_files, output_valid_files, output_test_files, valid_rate=0.2):
    label_list = []
    with open(input_files[1]) as f:
        for line in f:
            label_list.append(line.split()[0])

    n = len(label_list)
    a = np.arange(n)
    np.random.shuffle(a)
    label_tag = dict()
    for i in a[0: int(n*valid_rate)]:
        label_tag[label_list[i]] = 0
    for i in a[int(n*valid_rate):]:
        label_tag[label_list[i]] = 1

    def reassint(input, output0, output1):
        with open(input) as fin, open(output0, 'wt') as fout0, open(output1, 'wt') as fout1:
            for line in fin:
                label = line.split()[0]

                if label in label_tag:
                    flag = label_tag[label]
                else:
                    i = label.rfind('-')
                    label = label[0:i]
                    flag = label_tag[label]

                if flag == 0:
                    fout0.write(line)
                else:
                    fout1.write(line)

    for src_file, out_valid, out_test in zip(input_files, output_valid_files, output_test_files):
        reassint(src_file, out_valid, out_test)


if __name__ == '__main__':

    #######################################
    #  run 'local/task.py' to get src data
    ######################################

    root_dir = '../data/pkurmrb'
    all_data = os.path.join(root_dir, 'data.txt')
    process_pku_data(os.path.join(root_dir, 'pku_rmrb_1998_2000.ci.utf8'), all_data)

    # count text
    txtinfo = wb.TxtInfo(all_data)
    txtinfo.nWord /= 2
    print(txtinfo)

    # split into the train, valid, test
    pku_train = os.path.join(root_dir, 'train.txt')
    pku_valid = os.path.join(root_dir, 'valid.txt')
    pku_test = os.path.join(root_dir, 'test.txt')
    np.random.seed(0)
    split_to_train_valid_test(all_data, pku_train, pku_valid, pku_test, valid_rate=0.01, test_rate=0.01)

    # hkust data
    hkust_root = wb.mkdir('../data/hkust_char/')
    hkust_train = os.path.join(hkust_root, 'train')
    hkust_valid = os.path.join(hkust_root, 'valid')
    process_hkust_data(os.path.join('../data/train'), hkust_train)
    process_hkust_data(os.path.join('../data/dev'), hkust_valid)

    process_hkust_data(os.path.join('../data/train'), hkust_train + '.chr', add_pos=False)
    process_hkust_data(os.path.join('../data/dev'), hkust_valid + '.chr', add_pos=False)

    # create hkust vocabulary
    data = seq.DataX(total_level=2,
                     train_list=[hkust_train],
                     valid_list=[hkust_valid],
                     )
    hkust_vocab_chr = os.path.join(hkust_root, 'vocab.chr')
    hkust_vocab_pos = os.path.join(hkust_root, 'vocab.pos')
    data.vocabs[0].write(hkust_vocab_chr)
    data.vocabs[1].write(hkust_vocab_pos)

    # hkust nbest
    src_nbests = ['../data/nbest/' + s for s in ['words_text', 'test_filt.txt', 'acwt', 'lmwt.lmonly', 'lmwt.nolm']]
    nbest_names = ['nbest', 'trans', 'acwt', 'lmwt.lmonly', 'lmwt.nolm']
    nbest_valid_files = [wb.mkdir('../data/nbest/valid/') + s for s in nbest_names]
    nbest_test_files = [wb.mkdir('../data/nbest/test/') + s for s in nbest_names]
    np.random.seed(0)
    split_nbest_to_valid_test(src_nbests, nbest_valid_files, nbest_test_files, valid_rate=0.2)

    nbest_chr = [wb.mkdir(hkust_root + 'nbest/') + s for s in nbest_names]
    nbest_valid_chr = [wb.mkdir(hkust_root + 'nbest/valid/') + s for s in nbest_names]
    nbest_test_chr = [wb.mkdir(hkust_root + 'nbest/test/') + s for s in nbest_names]
    nbest_to_char(src_nbests, nbest_chr)
    nbest_to_char(nbest_valid_files, nbest_valid_chr)
    nbest_to_char(nbest_test_files, nbest_test_chr)

    infos = {'pku_train': [pku_train],
             'pku_valid': [pku_valid],
             'pku_test': [pku_test],
             'hkust_train': [hkust_train],
             'hkust_valid': [hkust_valid],
             'hkust_train_chr': [hkust_train + '.chr'],
             'hkust_valid_chr': [hkust_valid + '.chr'],
             'hkust_train_wod': ['../data/train'],
             'hkust_valid_wod': ['../data/dev'],
             'hkust_vocab_chr': hkust_vocab_chr,
             'hkust_vocab_pos': hkust_vocab_pos,
             'nbest_chr': tuple(nbest_chr),
             'nbest_valid_chr': tuple(nbest_valid_chr),
             'nbest_test_chr': tuple(nbest_test_chr),
             'nbest_wod': tuple(src_nbests),
             'nbest_valid_wod': tuple(nbest_valid_files),
             'nbest_test_wod': tuple(nbest_test_files)
             }

    with open('data.info', 'wt') as f:
        json.dump(infos, f, indent=4)





