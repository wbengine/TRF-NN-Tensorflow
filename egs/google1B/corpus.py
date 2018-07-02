import os
import sys
import json
import glob

sys.path.insert(0, os.getcwd() + '/../../tfcode/')
from base import *


def get_train_text(datadir, fwrite, num):
    print('write word:', fwrite)
    a = os.listdir(datadir)
    files = sorted([datadir + x for x in a if os.path.isfile(datadir + x)])

    with open(fwrite, 'wt') as wf:
        for i in range(min(num, len(files))):
            print('  [add] ', files[i])
            with open(files[i], 'rt', errors='ignore') as f:
                for line in f:
                    wf.write(line.lower())


def get_vocab(fread, init_vocab={}, cutoff=0, maxnum=None):
    """
    get the vocabulary
    Args:
        fread: a file name, or a str with '*.' or a list of file name
        init_vocab: the init vocabulary
        cutoff: the cotoff setting, the word with number less than cutoff will be cut
        maxnum: the maximum vocabulary size

    Returns:
        dict()
    """
    v = init_vocab

    if fread is None:
        file_list = []
    elif isinstance(fread, list):
        file_list = fread
    elif isinstance(fread, str):
        file_list = glob.glob(fread)
    else:
        raise TypeError('Unknown input parameter fread={}'.format(fread))

    print('get_vocab...')
    for fname in file_list:
        print('\tprocess:', fname)
        with open(fname, 'rt', errors='ignore') as f:
            for line in f:
                for w in line.split() + ['<s>', '</s>']:
                    v.setdefault(w.lower(), 0)
                    v[w.lower()] += 1

    # cutoff
    if cutoff > 1:
        print('\tcutoff={}...'.format(cutoff))
        v_final = dict()
        for w, n in v.items():
            if n >= cutoff:
                v_final[w] = n
        v = v_final

    if maxnum is not None:
        print('\tmax-size={}...'.format(maxnum))
        v_final = dict()
        wlist = sorted(list(v.items()), key=lambda x: x[1], reverse=True)
        for w, n in wlist[0:maxnum]:
            v_final[w] = n
        v = v_final

    # add unk
    if '<unk>' not in v:
        v['<unk>'] = 1

    return v


def write_vocab(fname, v, is_sorted=True, write_info=('word', 'count')):
    """
    write the vocabulary to files
    Args:
        fname: the file name
        v: dict(), the vocabulary, v[w] = the unigram count
        is_sorted: True (default), then sorted the v based the count in dict
        write_info: tuple. 'id' is the word-id, 'count' is the unigram count, 'word' is the word string

    Returns:
        None
    """
    if is_sorted:
        wlist = sorted(list(v.items()), key=lambda x: x[1], reverse=True)
    else:
        wlist = v.items()

    with open(fname, 'wt') as f:
        for i, (w, n) in enumerate(wlist):
            a = []
            for slot in write_info:
                if slot == 'word':
                    a.append(w)
                elif slot == 'id':
                    a.append(i)
                elif slot == 'count':
                    a.append(n)
                else:
                    raise TypeError('unknown slot = {}'.format(slot))

            f.write('\t'.join(['{}']*len(a)).format(*a))
            f.write('\n')


def get_text(fread, fwrite, vocab=None, unk='<unk>'):
    # remove the unicode character
    print('write word:', fwrite)
    with open(fread, 'rt', errors='ignore') as rf, open(fwrite, 'wt') as wf:
        for line in rf:
            line = line.lower()
            if vocab is not None:
                wlist = []
                for w in line.split():
                    if w in vocab:
                        wlist.append(w)
                    else:
                        wlist.append(unk)
                wf.write(' '.join(wlist) + '\n')
            else:
                wf.write(line)


def word_text_to_char_text(fread, fwrite, is_nbest=False):
    print('write char:', fwrite)
    with open(fread, 'rU') as rf, open(fwrite, 'wt') as wf:
        for line in rf:
            if is_nbest:
                s = '_'.join(line.split()[1:])
            else:
                s = '_'.join(line.split())
            wf.write(' '.join(list(s)) + '\n')
            
raw_word_datas = ['./data/' + s + '.txt' for s in ['train', 'valid', 'test']]
raw_char_datas = ['./data/' + s + '.char.txt' for s in ['train', 'valid', 'test']]


def char_raw_dir():
    return tuple(raw_char_datas)

if wb.is_window():
    dataroot = 'Z:\\wangb\\Data\\1-billion-word-language-modeling-benchmark-r13output\\'
else:
    dataroot = '/home/wangbin/NAS_workspace/wangb/Data/1-billion-word-language-modeling-benchmark-r13output/'
traindir = dataroot + 'training-monolingual.tokenized.shuffled/'
train_txt = traindir + 'news.en-*-of-00100'
valid_txt = dataroot + 'heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050'
test_txt = dataroot + 'heldout-monolingual.tokenized.shuffled/news.en.heldout-00001-of-00050'


def word_raw_dir(max_files=1000):
    trains = glob.glob(train_txt)[0: max_files]
    return trains, valid_txt, test_txt


def extract_1billion_vocab():
    tskdir = './data/'
    wb.mkdir(tskdir)

    word_count = 'data/vocab_count_all.txt'
    if os.path.exists(word_count):
        v_all = dict()
        with open(word_count, 'rt') as f:
            for line in f:
                a = line.split()
                v_all[a[0]] = int(a[1])
    else:
        v_all = get_vocab(train_txt)
        write_vocab(word_count, v_all, write_info=('word', 'count'))
    print('all vocab={}', len(v_all))

    info = dict()
    for cutoff in range(0, 21):
        v = get_vocab(None, v_all, cutoff)
        print('vocab cutoff={} vocab={}'.format(cutoff, len(v)))
        file = 'data/vocab_cutoff{}.txt'.format(cutoff)
        write_vocab(file, v, is_sorted=True, write_info=('id', 'word', 'count'))
        info['vocab_cutoff{}'.format(cutoff)] = '../' + file

    for size in [10000, 20000, 50000]:
        v = get_vocab(None, v_all, maxnum=size)
        print('vocab max_size vocab={}'.format(size, len(v)))
        file = 'data/vocab_{}K.txt'.format(size // 1000)
        write_vocab(file, v, is_sorted=True, write_info=('id', 'word', 'count'))
        info['vocab_{}K'.format(size // 1000)] = '../' + file

    return info

if __name__ == '__main__':
    info = dict()
    # info = extract_1billion_vocab()

    info['train_all'], info['valid'], info['test'] = word_raw_dir()

    wb.mkdir('data')
    v = vocab.Vocab()
    for cutoff in [3, 5, 10]:
        v.generate_vocab(info['train_all'], cutoff=cutoff,
                         add_beg_token='<s>',
                         add_end_token='</s>',
                         add_unk_token='<unk>',
                         to_lower=True)
        print('vocab max_size vocab={}'.format(v.get_size()))
        v.write('data/vocab_cutoff{}.txt'.format(cutoff))
        info['vocab_cut%d' % cutoff] = '../data/vocab_cutoff{}.txt'.format(cutoff)

    with open('data.info', 'wt') as f:
        json.dump(info, f, indent=4)



