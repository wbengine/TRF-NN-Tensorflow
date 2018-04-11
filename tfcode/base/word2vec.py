# import tensorflow as tf
import os
import numpy as np

# the binary dir for word2vec tools
# bindir = '/mnt/workspace/wangbin/work/TRF-NN-tensorflow/tools/word2vec/c/'
bindir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../tools/word2vec/c/')


def word2vec(finput, foutput, dim):
    """
    The input files should not contain </s> tokens,
    but should contain <s> if you need the embedding of the beginning token
    """
    cmd = bindir + 'word2vec'
    cmd += ' -train \"%s\"' % finput
    cmd += ' -output \"%s\"' % foutput
    cmd += ' -size %d' % dim
    cmd += ' -cbow 0 -window 7 -negative 1 -hs 1 -sample 1e-3 -threads 12 -binary 0'
    os.system(cmd)


def word_cluster(finput, foutput, cnum, dim=200):
    """
    The input files should not contain </s> tokens,
    but should contain <s> if you need the embedding of the beginning token
    """
    cmd = bindir + 'word2vec'
    cmd += ' -train ' + finput
    cmd += ' -output ' + foutput
    cmd += ' -size {}'.format(dim)
    cmd += ' -classes {}'.format(cnum)
    cmd += ' -cbow 0 -window 7 -negative 1 -hs 1 -sample 1e-3 -threads 12 -binary 0'
    os.system(cmd)


def read_vec(fvectors, word_to_id=None, random_initial=0.1):
    """
    Args:
        fvectors: the vector file
        word_to_id: the vocabulary mapping str to id
        random_initial: if the word is not observed in fvectors file,
            then random the embedding using this values

    Returns:

    """
    with open(fvectors, 'rt') as f:
        head = f.readline().split()
        vocab_size = int(head[0])
        vec_dim = int(head[1])

        if word_to_id is not None:
            true_vocab_size = len(word_to_id)
            if true_vocab_size != vocab_size:
                print('\n[W] vocab_size({}) in files is not '
                      'equal to the vocab_size({}) in word-to-id'.format(
                        vocab_size, true_vocab_size))
        else:
            true_vocab_size = vocab_size

        buf = np.random.uniform(low=-random_initial,
                                high=random_initial,
                                size=(true_vocab_size, vec_dim))
        for i in range(vocab_size):
            a = f.readline().split()
            if a[0].isalnum():
                wid = int(a[0])
            else:
                wid = word_to_id[a[0]]
            emb = np.array([float(i) for i in a[1:]])
            assert len(emb) == vec_dim
            buf[wid] = emb
    return buf


def write_vec(fvectors, buf):
    with open(fvectors, 'wt') as f:
        vocab_size = buf.shape[0]
        vec_dim = buf.shape[1]
        f.write('{} {}\n'.format(vocab_size, vec_dim))
        for (wid, emb) in enumerate(buf):
            f.write(str(wid) + '\t')
            f.write(' '.join([str(i) for i in emb]) + '\n')



