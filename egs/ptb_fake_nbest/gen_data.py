import os

from base import *


output_dir = 'data'
train = '../ptb_wsj0/data/ptb/ptb.train.txt'
valid = '../ptb_wsj0/data/ptb/ptb.valid.txt'
test = '../ptb_wsj0/data/ptb/ptb.test.txt'
train_files = [train, valid, test]
train_nbest_files = [output_dir + '/train_nbest.txt', output_dir + '/train_trans.txt']
test_nbest_files = [output_dir + '/test_nbest.txt', output_dir + '/test_trans.txt']


def generate_vocab(seq_list):
    v = dict()
    for x in seq_list:
        for w in x:
            v[w] = 0

    word_list = []
    for w in v.keys():
        word_list.append(w)

    return word_list


# generate fake nbest
def generate_fake_nbest(seq_list, add_self=False):

    word_list = generate_vocab(seq_list)

    nbest_list = []
    for x in seq_list:
        nbest_for_x = []
        for i in range(1, 5):
            a = [generate_errors(x, word_list, ['sub', 'ins', 'del'], i) for _ in range(10)]
            nbest_for_x += a
        if add_self:
            nbest_for_x.append(list(x))
        nbest_list.append(nbest_for_x)

    return nbest_list


def generate_errors(x, word_list, types, num):
    """
    :param x: a list
    :param word_list: a list of word
    :param types: error types, 'sub', 'ins', 'del'
    :param num: error numbers
    :return: a error list
    """
    error_x = list(x)
    vocab_size = len(word_list)
    for _ in range(num):
        error_type = np.random.choice(types)
        if error_type == 'sub':
            i = np.random.randint(len(error_x))
            w = word_list[np.random.randint(vocab_size)]
            error_x[i] = w
        elif error_type == 'ins':
            i = np.random.randint(len(error_x) + 1)
            w = word_list[np.random.randint(vocab_size)]
            error_x.insert(i, w)
        elif error_type == 'del':
            if len(error_x) == 1:
                continue
            i = np.random.randint(len(error_x))
            del error_x[i]
        else:
            raise TypeError('undefined error type=' + error_type)

    return error_x


def main_test():
    wb.mkdir(output_dir)

    # read test
    print('read test sequences')
    test_seq_list = []
    test_label_list = []
    with open(test, 'rt') as f:
        for i, s in enumerate(f):
            test_seq_list.append(s.split())
            test_label_list.append('ptbtest{:0>4}'.format(i))

    # gen nbest
    print('generate nbest')
    test_nbest_list = generate_fake_nbest(test_seq_list, add_self=False)
    with open(test_nbest_files[0], 'wt') as f:
        for label, nbest in zip(test_label_list, test_nbest_list):
            for i, seq in enumerate(nbest):
                f.write('{}-{} {}\n'.format(label, i+1, ' '.join(seq)))

    # gen transcript
    print('write transcript')
    with open(test_nbest_files[1], 'wt') as f:
        for label, seq in zip(test_label_list, test_seq_list):
            f.write('{} {}\n'.format(label, ' '.join(seq)))


def main_train():
    wb.mkdir(output_dir)
    # training sequences
    print('read train sequences')
    test_seq_list = []
    test_label_list = []
    with open(train, 'rt') as f:
        for i, s in enumerate(f):
            test_seq_list.append(s.split())
            test_label_list.append('ptbtrain{:0>4}'.format(i))

    nbest_num = 3000
    a = np.arange(len(test_seq_list))
    np.random.shuffle(a)
    a = a[0: nbest_num]
    test_seq_list = [test_seq_list[i] for i in a]
    test_label_list = [test_label_list[i] for i in a]

    # gen nbest
    print('generate nbest')
    test_nbest_list = generate_fake_nbest(test_seq_list, add_self=True)
    with open(train_nbest_files[0], 'wt') as f:
        for label, nbest in zip(test_label_list, test_nbest_list):
            for i, seq in enumerate(nbest):
                f.write('{}-{} {}\n'.format(label, i + 1, ' '.join(seq)))

    # gen transcript
    print('write transcript')
    with open(train_nbest_files[1], 'wt') as f:
        for label, seq in zip(test_label_list, test_seq_list):
            f.write('{} {}\n'.format(label, ' '.join(seq)))


nbest_real = reader.NBest(*reader.wsj0_nbest())
nbest_real_lmonly = reader.NBest(*reader.wsj0_nbest())
nbest_real_lmonly.acscore = None
nbest_fake_train = reader.NBest(*train_nbest_files)
nbest_fake_test = reader.NBest(*test_nbest_files)


def nbest_eval(m, data, workdir, fres, res_name, rescore_fun=None):

    def rescore(seq_list):
        if rescore_fun is not None:
            return rescore_fun(seq_list)
        else:
            return m.rescore(seq_list)

    # write nbest list
    nbest_real.write_nbest_list(os.path.join(workdir, 'nbest.real.id'), data)
    nbest_fake_test.write_nbest_list(os.path.join(workdir, 'nbest.fake.test.id'), data)
    nbest_fake_train.write_nbest_list(os.path.join(workdir, 'nbest.fake.train.id'), data)
    save_lmscore_dir = wb.mkdir(os.path.join(workdir, 'lmscores'))

    with wb.processing('rescoring_nbest_real'):
        nbest_real.lmscore = rescore(nbest_real.get_nbest_list(data))
    nbest_real.write_lmscore(os.path.join(workdir, 'nbest.lmscore'))
    nbest_real.write_lmscore(os.path.join(save_lmscore_dir, res_name + '.lmscore'))
    wer_real = nbest_real.wer()
    nbest_real.write_log(os.path.join(workdir, 'wer.log'))
    # print('wer={} lmscale={}, acscale={}'.format(wer_real, nbest_real.lmscale, nbest_real.acscale))
    fres.Add(res_name, ['WER', 'lmscale'], [wer_real, nbest_real.lmscale])

    nbest_real_lmonly.lmscore = nbest_real.lmscore
    wer_lmonly = nbest_real_lmonly.wer(lmscale=[1])
    fres.Add(res_name, ['lmonly-wer'], [wer_lmonly])

    wer_fakes = []
    for nbest_fake, fake_name in zip([nbest_fake_train, nbest_fake_test], ['train', 'test']):
        nbest_fake.lmscore = rescore(nbest_fake.get_nbest_list(data))
        nbest_fake.write_lmscore(os.path.join(workdir, 'nbest.fake.%s.lmscore' % fake_name))
        wer = nbest_fake.wer(lmscale=[1])
        wer_fakes.append(wer)

    fres.Add(res_name, ['train-fake-wer', 'test-fake-wer'], wer_fakes)

    return wer_real, wer_lmonly, wer_fakes


if __name__ == '__main__':
    np.random.seed(0)

    main_train()
    main_test()













