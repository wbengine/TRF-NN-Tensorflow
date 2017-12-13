import os
import collections
import numpy as np
from sklearn.cluster import KMeans

# import tensorflow as tf
from . import wblib as wb
from . import word2vec


class Data(object):
    def __init__(self):
        self.beg_token_str = None
        self.end_token_str = None
        self.unk_token_str = None
        self.rm_beg_end_in_data = False
        self.word_to_id = dict()
        self.word_to_class = None
        self.word_count = dict()
        self.word_list = []
        self.datas = []

        # character vocabulary
        self.char_to_id = dict()
        self.char_list = []
        self.word_to_chars = []

    def load_raw_data(self, file_list, add_beg_token=None, add_end_token='</s>',
                      add_unknwon_token='<unk>', min_length=None, max_length=None, reverse_sentence=False,
                      rm_beg_end_in_data=False):
        """
        load raw data form txt files

        Args:
            file_list: a list of filenames, the fist is used to build vocabulary
            add_beg_token: string or None. If not None, then add begin token.
            add_end_token: string or None. If not None, then add end token.
            add_unknwon_token: if not None, then add unkown token to vocabulary
            min_length: if not None, then just load the line with length >= min_length
            max_length: if not None, then just load the line with length <= max_length
            reverse_sentence: if True, then reverse the each sentence, used to training reversed-LSTM
            rm_beg_end_in_data: if True, then the read data lists donot containing the <s> and </s>

        Returns:
            self pointer

        """
        print('reader.Data: load raw data...')
        self.beg_token_str = add_beg_token
        self.end_token_str = add_end_token
        self.unk_token_str = add_unknwon_token
        self.rm_beg_end_in_data = rm_beg_end_in_data

        raw_datas = []
        for filename in file_list:
            data = _read_sequences(filename, add_beg_token, add_end_token)
            if min_length is not None or max_length is not None:
                total_lines = len(data)
                if max_length is not None:
                    data = list(filter(lambda x: len(x) <= max_length, data))
                if min_length is not None:
                    data = list(filter(lambda x: len(x) >= min_length, data))
                reset_lines = len(data)
                print('maxlen={} remove {:.2f}% ({}) lines.'.format(max_length,
                                                                    (total_lines-reset_lines)/total_lines*100,
                                                                    (total_lines-reset_lines)))
            if reverse_sentence:
                for d in data:
                    d.reverse()
            raw_datas.append(data)

        self.word_to_id, self.word_count = _build_vocab(raw_datas,
                                                        add_unknown_token=add_unknwon_token,
                                                        add_beg_token=add_beg_token,
                                                        add_end_token=add_end_token)
        self.word_list = sorted(self.word_to_id.keys(), key=lambda x: self.word_to_id[x])
        self.build_char_vocab()

        # read datas
        skip_head = 0
        skip_tail = 0
        if self.rm_beg_end_in_data:
            # remove the beg end tokens in datas
            skip_head = self.beg_token_str is not None
            skip_tail = self.end_token_str is not None

        self.datas = [_file_to_word_ids(d, self.word_to_id, self.unk_token_str, skip_head, skip_tail) for d in raw_datas]

        return self

    def build_char_vocab(self, add_beg_end_tokens=True):
        self.char_to_id = dict()
        self.char_list = []
        for w in self.word_list:
            for c in w:
                self.char_to_id[c] = 0

        if not add_beg_end_tokens:  # do not add the beg/end tokens
            for i, c in enumerate(sorted(self.char_to_id.keys())):
                self.char_list.append(c)
                self.char_to_id[c] = i

            self.word_to_chars = []
            for w in self.word_list:
                self.word_to_chars.append([self.char_to_id[c] for c in w])
        else:
            # ids 0/1 is for beg/end tokens
            self.char_list.append('<s>')
            self.char_list.append('</s>')
            for i, c in enumerate(sorted(self.char_to_id.keys())):
                self.char_list.append(c)
                self.char_to_id[c] = i + 2

            self.word_to_chars = []
            for w in self.word_list:
                # add beg/end tokens
                self.word_to_chars.append([0] + [self.char_to_id[c] for c in w] + [1])

    def get_char_size(self):
        return len(self.char_list)

    def load_data(self, fread, is_nbest=False, max_length=None, reversed_sentence=False):
        if isinstance(fread, str):
            data = _read_sequences(fread, self.beg_token_str, self.end_token_str)
        else:
            data = fread
        if is_nbest:
            if self.beg_token_str is not None:
                data = [seq[0:1] + seq[2:] for seq in data]
            else:
                data = [seq[1:] for seq in data]

        if max_length is not None:
            total_line = len(data)
            data = list(filter(lambda x: len(x) <= max_length, data))
            result_line = len(data)

            print('[Data] load_data, length filter remove {}({:.2f}\%) lines'.format(
                total_line-result_line,
                100*(total_line-result_line)/total_line))

        if reversed_sentence:
            for d in data:
                d.reverse()

        # read datas
        skip_head = 0
        skip_tail = 0
        if self.rm_beg_end_in_data:
            # remove the beg end tokens in datas
            skip_head = self.beg_token_str is not None
            skip_tail = self.end_token_str is not None
        data_id = _file_to_word_ids(data, self.word_to_id, self.unk_token_str, skip_head, skip_tail)
        return data_id

    def write_vocab(self, fwrite):
        with open(fwrite, 'wt') as f:
            for i, w in enumerate(self.word_list):
                if self.word_to_chars:
                    f.write('{}\t{}\t{}\n'.format(i, w, '-'.join([str(k) for k in self.word_to_chars[i]])))
                else:
                    f.write('{}\t{}\n'.format(i, w))

    def read_vocab(self, fread):
        v = dict()
        vlist = []
        with open(fread, 'rt') as f:
            for line in f:
                a = line.split()
                i = int(a[0])
                w = a[1]
                v[w] = i
                vlist.append(w)
                assert len(vlist) == i + 1
        return v, vlist

    def write_char_vocab(self, fwrite, fword2chars=None):
        with open(fwrite, 'wt') as f:
            for i, c in enumerate(self.char_list):
                f.write('{}\t{}\n'.format(i, c))
        if fword2chars is not None:
            with open(fword2chars, 'wt') as f:
                for i, cs in enumerate(self.word_to_chars):
                    f.write('{}\t{}\t{}\t{}\n'.format(i,
                                                  self.word_list[i],
                                                  ' '.join([str(n) for n in cs]),
                                                  ''.join([self.char_list[n] for n in cs])
                                                  ))

    def write_data(self, seq_list, write_file,
                   skip_beg_token=False, skip_end_token=False, write_text=False):
        """
        write data to file
        Args:
            seq_list: a list of sequence (id list)
            write_file: write file
            skip_beg_token: if True then remove the beg-token
            skip_end_token: if True then remove the end-token
            write_text: if True then write the text instead of the token-id

        Returns:
            None
        """
        if isinstance(seq_list, int):
            write_data = self.datas[seq_list]
        else:
            write_data = seq_list
        # write
        with open(write_file, 'wt') as f:
            for wid in write_data:
                write_ids = list(wid)
                if skip_beg_token and self.beg_token_str is not None:
                    del write_ids[0]

                if skip_end_token and self.end_token_str is not None:
                    del write_ids[-1]

                if write_text:
                    f.write(' '.join([self.word_list[i] for i in write_ids]) + '\n')
                else:
                    f.write(' '.join([str(i) for i in write_ids]) + '\n')

    def print_data(self, seq_list, name='dataset'):
        """
        print the corpus information
        including the token number and sentence number.
        """
        if isinstance(seq_list, int):
            seq_list = self.datas[seq_list]

        s = name + '\t'
        s += 'tokens={} \t sents={}'.format(*_count_data_info(seq_list))
        print(s)

    def get_max_len(self):
        lens = [len(line) for line in self.datas[0]]
        maxlen = np.max(lens)
        return maxlen

    def get_vocab_size(self):
        return len(self.word_list)

    def get_beg_token(self):
        try:
            return self.word_to_id[self.beg_token_str]
        except KeyError:
            return None

    def get_end_token(self):
        try:
            return self.word_to_id[self.end_token_str]
        except KeyError:
            return None

    def get_unk_token(self):
        try:
            return self.word_to_id[self.unk_token_str]
        except KeyError:
            return None

    def get_min_len(self):
        min_len = 1
        if self.beg_token_str is not None and not self.rm_beg_end_in_data:
            min_len += 1
        if self.end_token_str is not None and not self.rm_beg_end_in_data:
            min_len += 1
        # lens = [len(line) for line in self.datas[0]]
        # min_len = np.min(lens)
        return min_len

    def get_pi_true(self, data=None):
        """
        get the length distribution of a given data

        Args:
            data: a list of lists

        Returns:
            a np.array: the length distribution
        """
        if data is None:
            data = self.datas[0]  # the default training set

        lens = [len(line) for line in data]
        pi = np.zeros(max(lens) + 1)
        for l in lens:
            pi[l] += 1
        pi /= np.sum(pi)

        # remove zero probs
        if np.min(pi) <= 1e-8:
            pi[self.get_min_len():] += 1e-5
            pi /= np.sum(pi)

        return pi

    def get_pi0(self, pi_true=None, add=None):
        if pi_true is None:
            pi_true = self.get_pi_true()
        pi0 = np.array(pi_true)
        idx = np.argmax(pi0)

        min_len = self.get_min_len()

        if add is None:
            pi0[min_len:idx+1] = pi0[idx]
        else:
            pi0[min_len:idx+1] += add
        pi0[0: min_len] = 0
        pi0 /= np.sum(pi0)
        return pi0

    def get_unigram(self):
        unigram = np.zeros(self.get_vocab_size())
        for seq in self.datas[0]:
            for i in seq:
                unigram[i] += 1
        unigram /= unigram.sum()
        return unigram

    def get_bigram(self):
        with wb.processing('create bigram'):
            context = dict()
            for seq in self.datas[0]:
                for i in range(len(seq)-1):
                    h = seq[i]
                    w = seq[i+1]
                    if h not in context:
                        context[h] = np.zeros(self.get_vocab_size())
                    context[h][w] += 1
            # normalize
            for p in context.values():
                p /= np.sum(p)

        return context

    def get_w2c_map(self):
        return produce_data_to_trf(self.word_to_chars)

    def word2vec(self, fvectors, dim, cnum=200):
        """
        performing word2vec and output the vector files
        """
        fdata = fvectors + '.srcdata'
        self.write_data(0, fdata, skip_end_token=True)
        word2vec.word2vec(fdata, fvectors + '.temp', dim)
        e = word2vec.read_vec(fvectors + '.temp', self.word_to_id)
        word2vec.write_vec(fvectors, e)  # to remove the </s> token
        os.remove(fvectors + '.srcdata')
        os.remove(fvectors + '.temp')

        # using e to cluster the word
        if cnum > 0:
            cluster_file = fvectors + '.c{}.cluster'.format(cnum)
            if wb.exists(cluster_file):
                print("load the [exist] cluster file:", cluster_file)
                self.cluster_data_io(cluster_file, 'read')
            else:
                print('k-means cluster, c={}'.format(cnum))
                kmeans = KMeans(n_clusters=cnum, random_state=0).fit(e)
                self.word_to_class = kmeans.labels_
                self.cluster_data_io(cluster_file, 'write')

    def cluster_data_io(self, cluster_file, read_or_write):
        if read_or_write == 'read':
            print("load the [exist] cluster file:", cluster_file)
            self.word_to_class = np.zeros(self.get_vocab_size(), dtype='int')
            with open(cluster_file, 'rt') as f:
                for line in f:
                    a = line.split()
                    self.word_to_class[int(a[0])] = int(a[2].split('=')[-1])
        else:
            print('write cluster to file:', cluster_file)
            with open(cluster_file, 'wt') as f:
                for i, (w, c) in enumerate(zip(self.word_list, self.word_to_class)):
                    f.write('{}\t{}\tclass={}\n'.format(i, w, c))

    def seqs_to_class(self, seq_list):
        if self.word_to_class is None:
            raise TypeError('Unkown word to class information. Run data().word2vec')
        return [self.word_to_class[np.array(s)].tolist() for s in seq_list]

    def seqs_to_text(self, seq_list, skip_beg_token=False, skip_end_token=False):
        """[[1,2,3]] ==> [[This is good]]"""
        def seq_filter(s):
            if skip_beg_token and self.beg_token_str is not None:
                s = s[1:]
            if skip_end_token and self.end_token_str is not None:
                s = s[0: -1]
            return s
        return [[self.word_list[i] for i in seq_filter(seq)] for seq in seq_list]

    def cut_data_to_length(self, seqs, maxlen, minlen=None):
        """
        split each sequence to a set of short sequences
        The return is (res_list, res_index)
        The input sequence seqs[i] is splited to
           res_list[res_index[i][0]: res_index[i][1]]
        Args:
            seqs: a list of sequence or a sequence
            maxlen: the maximum length, including the beg/end tokens

        Returns:
            (res_list, res_index)
        """
        beg = 1 if self.beg_token_str is not None else 0
        end = 1 if self.end_token_str is not None else 0
        if self.rm_beg_end_in_data:
            beg = 0
            end = 0
        maxlen -= beg + end

        if minlen is None:
            minlen = 1
        else:
            minlen -= beg + end

        if isinstance(seqs[0], int):
            seqs = [seqs]

        res_list = []
        res_index = []
        for seq in seqs:
            idx_1 = len(res_list)

            seq = list(seq)

            # split
            a = split_sequence_to_range(seq[beg: len(seq)-end], minlen, maxlen)
            for x in a:
                res_list.append(seq[0:beg] + x + seq[-end:])
            # for i in range(beg, len(seq)-end, maxlen):
            #     a = seq[0:beg] + seq[i: min(i+maxlen, len(seq)-end)] + seq[len(seq)-end:]
            #     res_list.append(a)
            idx_2 = len(res_list)
            res_index.append((idx_1, idx_2))

        return res_list, res_index

    def cut_train_to_length(self, maxlen):
        self.test_max_len(maxlen)
        self.datas[0], _ = self.cut_data_to_length(self.datas[0], maxlen)

    def test_max_len(self, maxlen):
        # output the precent of sentences longer than maxlen
        for i in range(len(self.datas)):
            lens = [len(x) for x in self.datas[i]]
            longer_num = len(list(filter(lambda x: x>maxlen, lens)))
            print('reader.Data.datas[{}]: {:,} ({:.2f}\%) sentences longer than {}'.format(
                i, longer_num, 100 * longer_num / len(lens), maxlen))

    def rm_beg_end_token(self, seq, rm_beg=True, rm_end=True):
        new_seq = list(seq)
        if rm_beg and new_seq[0] == self.get_beg_token():
            del new_seq[0]

        if rm_end and new_seq[-1] == self.get_end_token():
            del new_seq[-1]

        return new_seq

    def rm_beg_tokens_in_datas(self):
        # remove all the beg-tokens in datas,
        # but preserve it in the vocabulary
        beg_token_id = self.get_beg_token()
        for d in self.datas:
            for s in d:
                if s[0] == beg_token_id:
                    del s[0]


class LargeData(Data):
    def __init__(self):
        super().__init__()
        self.train_file_list = []
        self.train_file_num = 0

        self.max_length = None
        self.reverse_sentence = False

    def dynamicly_load_raw_data(self, sorted_vocab_file, train_list, valid_file=None, test_file=None,
                                add_beg_token=None,
                                add_end_token='</s>',
                                add_unknwon_token='<unk>',
                                max_length=None, reverse_sentence=False):
        """
        init the data
        Args:
            sorted_vocab_file: the vocab should be sorted based the count in training set from large to little
            train_list:
            valid_file:
            test_file:
            add_beg_token:
            add_end_token:
            add_unknwon_token:
            max_length:
            reverse_sentence:

        Returns:

        """
        self.train_file_list = train_list
        self.train_file_num = 0
        self.max_length = max_length
        self.reverse_sentence = reverse_sentence

        self.beg_token_str = add_beg_token
        self.end_token_str = add_end_token
        self.unk_token_str = add_unknwon_token

        # read vocabulary
        self.word_to_id = dict()
        self.word_list = []
        self.word_count = []
        with open(sorted_vocab_file, 'rt') as f:
            for line in f:
                a = line.split()
                i = int(a[0])
                w = a[1]
                self.word_to_id[w] = i
                self.word_list.append(w)
                assert len(self.word_list) == i + 1
                if len(a) >= 3:
                    self.word_count.append(int(a[2]))

        self.datas = []
        for fname in [self.train_file_list[self.train_file_num], valid_file, test_file]:
            self.datas.append(
                self.load_data(fname,
                               max_length=self.max_length,
                               reversed_sentence=self.reverse_sentence)
            )
        # self.cut_train_to_length(self.max_length)
        self.train_file_num += 1
        return self

    def next_train(self):
        self.datas[0] = self.load_data(self.train_file_list[self.train_file_num % len(self.train_file_list)],
                                       max_length=self.max_length,
                                       reversed_sentence=self.reverse_sentence)
        # self.cut_train_to_length(self.max_length)
        self.train_file_num += 1

    def get_train_epoch(self):
        return self.train_file_num // len(self.train_file_list)

    def get_unigram(self):
        if not self.word_count:
            raise TypeError('[{}.LargeData] cannot find the count in the vocabulary!!'.format(__name__))

        print('[{}.LargeData] get the unigram for multiple corpus'.format(__name__))
        count = np.array(self.word_count)
        return count / np.sum(count)


def _count_data_info(data):
    """
    count the sentence number and token number in data

    Args:
        data: filename/a list of lists

    Returns:
        sentence number (int32), token number (int32)
    """
    if isinstance(data, str):
        data = _read_sequences(data, add_beg_token=None, add_end_token=None)

    lens = [len(x) for x in data]
    sent_num = len(data)
    token_num = sum(lens)
    return token_num, sent_num


def _file_to_word_ids(data, word_to_id, unk=None, skip_head=0, skip_tail=0):
    """
    Trans data to word-id

    Args:
        data: filename/a list of string lists
        word_to_id: a word to id dictionary
        unk: the unk-token string
        skip_head: skip the frist n wrods
        skip_tail: skip the last n wrods

    Returns:
        a list of id list
    """
    if isinstance(data, str):
        data = _read_sequences(data)

    data_id = []
    for seq in data:
        wid = []
        if skip_tail > 0:
            residual_seq = seq[skip_head: -skip_tail]
        else:
            residual_seq = seq[skip_head:]
        for w in residual_seq:
            if w in word_to_id:
                wid.append(word_to_id[w])
            elif unk in word_to_id:
                wid.append(word_to_id[unk])
            else:
                print('[ERROR] cannot not find the word:', w)
        data_id.append(wid)
    return data_id


def _read_sequences(filename, add_beg_token=None, add_end_token='</s>'):
    """
    read a txt file to a list of string lists.
    i.e. [['this' 'is' 'a' 'good' 'day'], ['it' 'is' 'ok'], ... ]

    Args:
        filename: the txt file name
        add_beg_token: string or None. If not None, then add begin token.
        add_end_token: string or None. If not None, then add end token.

    Returns:
        a list of string lists

    """
    with open(filename, 'rt', errors='ignore') as f:
        return [([add_beg_token] if add_beg_token is not None else []) +
                [w.lower() for w in line.split()] +
                ([add_end_token] if add_end_token is not None else [])
                for line in f]


def _build_vocab(seq_list_list, add_unknown_token=None, add_beg_token=None, add_end_token=None):
    """
    build a vocabulary from a given list of string lists

    Args:
        seq_list_list: several corpus
        add_unknown_token: if not None, add unknown_token to the vocabulary
        add_beg_token: str, if not None, then add and process the beg_token
        add_end_token: str, if not None, then add and process the end_token

    Returns:
        tuple (word_to_id, word_count)
    """
    counter = collections.Counter()
    for seq_list in seq_list_list:
        for seq in seq_list:
            counter.update(seq)

    # processing the counts of special tokens
    max_count = max([x[1] for x in counter.items()])
    if add_unknown_token is not None:
        counter[add_unknown_token] = max_count + 1
    if add_end_token is not None:
        counter[add_end_token] = max_count + 2
    if add_beg_token is not None:
        counter[add_beg_token] = max_count + 3

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, counts = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    word_count = dict(count_pairs)

    return word_to_id, word_count


def produce_data_for_rnn(raw_data, batch_size, step_size, include_residual_data=False):
    """
    concatenate the data and split to batches

    Args:
        raw_data: a list of id-lists
        batch_size: intger, batch size
        step_size: intger, the length of each subsequence.
        include_residual_data: is True, then the residual data in the tail of raw_data
            will alse all to the returned list (x, y).
            Note the residual is shorter than step_size

    Returns:
        tuple(
          a list of np.Array for inputs x,
          a list of np.Array for targets y)
    """
    one_data = []
    for d in raw_data:
        one_data += d

    one_data = np.array(one_data)
    data_len = np.size(one_data)
    batch_len = data_len // batch_size
    data = one_data[0: batch_size * batch_len].reshape([batch_size, batch_len]).astype('int32')

    epoch_size = (batch_len - 1) // step_size

    x_list = []
    y_list = []
    for i in range(epoch_size):
        x_list.append(data[:, i * step_size: (i + 1) * step_size])
        y_list.append(data[:, i * step_size + 1: (i + 1) * step_size + 1])

    # process residual data
    if include_residual_data and (batch_len - 1) % step_size > 0:
        x_list.append(data[:, epoch_size * step_size: -1])
        y_list.append(data[:, epoch_size * step_size + 1:])

    # if len(x_list) == 0 or len(y_list) == 0:
    #     raise TypeError('reader.producer return none data !!! ' +
    #                     '[epoch_size={}, include_residual_data={}]'.format(
    #                         epoch_size, include_residual_data)
    #                     )

    return x_list, y_list


def produce_data_for_rnn_len(raw_data, batch_size, step_size, pad_value=0):
    """
    generate data to batch size with length, such as
    inputs:
        raw_data = [[1,2,3,4,5], [1,2,3]],
        batch_size = 2
        step_size = 3
    return :
        x = [ [[1,2,3],[1,2,x]],
              [[4,x,x],[x,x,x]]
            ]
        y = [ [[2,3,4],[2,3,x]],
              [[5,x,x],[x,x,x]]]
        n = [ [3, 2],
              [1, 0] ]
    Args:
        raw_data: a list of sequences
        batch_size: integer
        step_size:  integer
        pad_value: integer

    Returns:
        a list of 2d array, a list of 2d array, a list of 1d array
    """
    x_list = []
    y_list = []
    n_list = []

    for i in range(0, len(raw_data), batch_size):
        d = raw_data[i: i+batch_size]
        if len(d) < batch_size:
            d += [[]] * (batch_size - len(d))

        data, lens = produce_data_to_array(d, pad_value)
        max_len = max(lens)
        pad_len = int(np.ceil((max_len-1) / step_size) * step_size + 1) - max_len
        data = np.pad(data, [[0, 0], [0, pad_len]], mode='constant', constant_values=pad_value)

        for j in range(0, max_len-1, step_size):
            x = data[:, j: j+step_size]
            y = data[:, j+1: j+step_size+1]
            n = np.minimum(np.maximum(0, lens-1-j), step_size)

            x_list.append(x)
            y_list.append(y)
            n_list.append(n)

    return x_list, y_list, n_list


# def producer_for_lm(raw_data, batch_size, num_steps, name=None, shuffle=False):
#     one_data = []
#     for x in raw_data:
#         one_data += x
#
#     with tf.name_scope(name, 'Producer_For_LM'):
#         raw_data = tf.convert_to_tensor(one_data, name='raw_data', dtype=tf.int32)
#
#         data_len = tf.size(raw_data)
#         batch_len = data_len // batch_size
#         data = tf.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])
#
#         epoch_size = (batch_len - 1) // num_steps
#         assertion = tf.assert_positive(
#             epoch_size,
#             message='epoch_size=0, decrease batch_size or num_steps')
#         with tf.control_dependencies([assertion]):
#             epoch_size = tf.identity(epoch_size, name='epoch_size')
#
#         i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
#         x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
#         x.set_shape([batch_size, num_steps])
#         y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
#         y.set_shape([batch_size, num_steps])
#
#         return x, y


def produce_data_to_array(seq_list, pad_value=0):
    """
    trans a list of sequence to a matrix, filling pad_value
    such as:
        [[1,2,3],
         [1,2] ]   ===>
        np.array([[1,2,3],
                 [1,2,x]])
        where x is the pad_value

    Args:
        seq_list:  a list of sequence
        pad_value: a int32

    Returns:
        tuple( np.array denoting the matrix,
               np.array denoting the length of each sequence )
    """
    max_len = np.max([len(x) for x in seq_list])
    num = len(seq_list)
    input_x = np.ones((num, max_len)) * pad_value
    input_n = np.array([len(x) for x in seq_list])
    for i in range(num):
        input_x[i][0: input_n[i]] = np.array(seq_list[i])

    return input_x.astype('int32'), input_n.astype('int32')


def extract_data_from_array(input_x, input_n=None):
    """trans [input_x, input_n] to seq_list"""
    if input_n is None:
        return [list(x) for x in input_x]
    else:
        return [list(x[0:n]) for x, n in zip(input_x, input_n)]


def produce_data_to_trf(seq_list, pad_value=0):
    return produce_data_to_array(seq_list, pad_value)


def extract_data_from_trf(input_x, input_n=None):
    return extract_data_from_array(input_x, input_n)


def merge_datas_for_trf(input_x_list, input_n_list, pad_value=0):
    """merge several (input_x, input_n) trf-data to one"""
    res_n = np.concatenate(input_n_list, axis=0)
    max_len = np.max(res_n)
    input_x_pad_list = [np.pad(input_x,
                               [[0, 0], [0, max_len-input_x.shape[1]]],
                               'constant', constant_values=pad_value)
                        for input_x in input_x_list]
    res_x = np.concatenate(input_x_pad_list, axis=0)
    return res_x, res_n


def produce_batch_seqs(seq_list, batch_size, output_precent=False):
    for i in range(0, len(seq_list), batch_size):
        if output_precent:
            yield seq_list[i: i+batch_size], min(1.0, (i + batch_size) / len(seq_list))
        else:
            yield seq_list[i: i+batch_size]


def produce_ngrams(seq_list, order):
    """produce the sequence list to ngrams"""
    for seq in seq_list:
        for i in range(len(seq) - order + 1):
            yield seq[i: i+order]


def split_sequence_to_range(seq, min_len, max_len):
    """split a list to several sequences of length ranging from mim_len to max_len"""
    a = []
    residual_seq = list(seq)
    residual_len = len(seq)
    while residual_len > 0:
        if residual_len > max_len:
            if residual_len - max_len >= min_len:
                a.append(residual_seq[0: max_len])
                residual_seq = residual_seq[max_len:]
            else:
                a.append(residual_seq[0:residual_len-min_len])
                a.append(residual_seq[-min_len:])
                residual_seq = []
        else:
            a.append(residual_seq)
            residual_seq = []
        residual_len = len(residual_seq)
    return a



def ptb_raw_dir():
    """PTB raw corpus"""
    file_path = os.path.split(os.path.realpath(__file__))[0]
    root = os.path.join(file_path, '../../egs/ptb_wsj0/data/ptb/')
    train = root + 'ptb.train.txt'
    valid = root + 'ptb.valid.txt'
    test = root + 'ptb.test.txt'
    return train, valid, test


def wsj0_nbest():
    """wsj0 nbest list and acscore"""
    abpath = os.path.split(os.path.realpath(__file__))[0]
    root = os.path.join(abpath, '../../egs/ptb_wsj0/data/WSJ92-test-data/')
    nbest = root + '1000best.sent'
    trans = root + 'transcript.txt'
    acscore = root + '1000best.acscore'
    lmscore = root + '1000best.lmscore'
    return nbest, trans, acscore, lmscore


def word_raw_dir():
    """PTB raw corpus"""
    abpath = os.path.split(os.path.realpath(__file__))[0]
    root = os.path.join(abpath, '../../egs/word/data/')
    train = root + 'train.words'
    valid = root + 'valid.words'
    test = root + 'test.words'
    return train, valid, test


def word_nbest():
    """word nbest data"""
    abpath = os.path.split(os.path.realpath(__file__))[0]
    root = os.path.join(abpath, '../../egs/word/data/')
    nbest = root + 'nbest.words'
    trans = root + 'transcript.words'
    return nbest, trans


class NBest(object):
    def __init__(self, nbest, trans, acscore=None, lmscore=None, gfscore=None):
        """
        construct a nbest class

        Args:
            nbest: nbet list
            trans: the test transcript ( correct text )
            acscore: acoustic score
            lmscore: language model score
            gfscore: graph score
        """
        self.nbest = nbest
        self.trans = trans
        self.acscore = None
        self.lmscore = None
        self.gfscore = gfscore
        if acscore is not None:
            self.acscore = wb.LoadScore(acscore)
        if lmscore is not None:
            self.lmscore = wb.LoadScore(lmscore)
        if gfscore is not None:
            self.gfscore = wb.LoadScore(gfscore)

        # save the best result
        self.lmscale = 1.0
        self.acscale = 1.0
        self.total_err = 0
        self.total_word = 0
        self.best_1best = None
        self.best_log = None

        self.nbest_list_id = None

    def process_best_file(self, best_file):
        new_best_file = wb.io.StringIO()
        for line in best_file:
            new_line = ' '.join(filter(lambda w: w.lower() != '<unk>', line.split()))
            new_best_file.write(new_line + '\n')
        best_file.close()
        new_best_file.seek(0)
        return new_best_file

    def wer(self, lmscale=np.linspace(0.1, 1.0, 10), rm_unk=False, sentence_process_fun=None):
        """
        compute the WER
        Returns:
            word error rate (WER)
        """
        if self.acscore is None:
            self.acscore = np.zeros(len(self.lmscore))
        if self.gfscore is None:
            self.gfscore = np.zeros(len(self.lmscore))

        # tune the lmscale
        opt_wer = 1000
        for ac in [1]:
            for lm in lmscale:
                s = ac * np.array(self.acscore) + lm * (np.array(self.lmscore) + np.array(self.gfscore))
                best_file = wb.io.StringIO()
                log_file = wb.io.StringIO()
                wb.GetBest(self.nbest, s, best_file)
                best_file.seek(0)

                if rm_unk:
                    best_file = self.process_best_file(best_file)

                [totale, totalw, wer] = wb.CmpWER(best_file, self.trans,
                                                  log_str_or_io=log_file,
                                                  sentence_process_fun=sentence_process_fun)

                # print('acscale={}\tlmscale={}\twer={}\n'.format(acscale, lmscale, wer))
                if wer < opt_wer:
                    opt_wer = wer
                    self.lmscale = lm
                    self.acscale = ac
                    self.total_word = totalw
                    self.total_err = totale

                    if self.best_1best is not None:
                        self.best_1best.close()
                    self.best_1best = best_file
                    self.best_1best.seek(0)

                    if self.best_log is not None:
                        self.best_log.close()
                    self.best_log = log_file
                    self.best_log.seek(0)

                else:
                    best_file.close()
                    log_file.close()

        return opt_wer

    def get_trans_txt(self, fwrite):
        # get the transcript text used to calculate PPL
        wb.file_rmlabel(self.trans, fwrite)

    def get_nbest_list(self, data):
        # return the nbest list id files used to rescoring
        if self.nbest_list_id is None:
            self.nbest_list_id = data.load_data(self.nbest, is_nbest=True)

            # process the empty sequences
            empty_len = int(data.beg_token_str is not None) + int(data.end_token_str is not None)
            for s in self.nbest_list_id:
                if len(s) == empty_len:
                    s.insert(-1, data.get_end_token())
        return self.nbest_list_id

    def write_lmscore(self, fwrite):
        with open(fwrite, 'wt') as fout, open(self.nbest, 'rt') as fin:
            for s, line in zip(self.lmscore, fin):
                fout.write(line.split()[0] + '\t' + str(s) + '\n')

    def write_log(self, fname):
        if self.best_log is None:
            print('[{0}.{1}] best_log=None, run {1}.wer() first.'.format(__name__, self.__class__.__name__))
        with open(fname, 'wt') as f:
            self.best_log.seek(0)
            f.write(self.best_log.read())

    def write_1best(self, fname):
        if self.best_1best is None:
            print('[{0}.{1}] best_1best=None, run {1}.wer() first.'.format(__name__, self.__class__.__name__))
        with open(fname, 'wt') as f:
            self.best_1best.seek(0)
            f.write(self.best_1best.read())


