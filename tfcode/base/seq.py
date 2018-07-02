import numpy as np

from base import *


class Seq(object):
    def __init__(self, input=10, level=2):
        # self.x[0] is the word sequence
        # self.x[1] is the level-1 hidden sequence
        if isinstance(input, int) or isinstance(input, np.int32) or isinstance(input, np.int64):
            # indicate the length
            self.x = np.zeros((level, input), dtype='int32')
        elif isinstance(input, list) or isinstance(input, np.ndarray):
            x = np.array(input, dtype='int32')
            if len(x.shape) == 1:
                x = np.reshape(x, [1, -1])
            if x.shape[0] > level:
                x = x[0: level]
            elif x.shape[0] < level:
                x = np.pad(x, [[0, level-x.shape[0]], [0, 0]], mode='constant')
            self.x = x
        else:
            raise TypeError('[{}.{}] unknown input type = {}'.format(
                __name__, self.__class__.__name__, type(input)))

    def get_level(self):
        return self.x.shape[0]

    def get_length(self):
        return self.x.shape[1]

    def get_seq(self, level=0):
        return self.x[level]

    def get_sub(self, beg_pos, end_pos=None):
        if end_pos is None:
            end_pos = self.get_length()
        elif end_pos < 0:
            end_pos += self.get_length()

        s = Seq(end_pos - beg_pos, level=self.get_level())
        s.x = self.x[:, beg_pos: end_pos]
        return s

    def random(self, min_len=3, max_len=10, vocab_sizes=10, beg_tokens=0, end_tokens=1, pi=None):
        if not isinstance(vocab_sizes, list):
            vocab_sizes = [vocab_sizes] * self.get_level()
        if not isinstance(beg_tokens, list):
            beg_tokens = [beg_tokens] * self.get_level()
        if not isinstance(end_tokens, list):
            end_tokens = [end_tokens] * self.get_level()

        if pi is None:
            seq_len = np.random.randint(min_len, max_len+1)
        else:
            pi[0:min_len] = 0
            pi /= pi.sum()
            seq_len = np.random.choice(max_len+1, p=pi)

        self.x = np.zeros((self.get_level(), seq_len), dtype='int32')
        for i in range(self.get_level()):
            self.x[i] = sp.random_seq(seq_len, seq_len, vocab_sizes[i], beg_tokens[i], end_tokens[i])

        return self

    def copy(self):
        new_seq = Seq()
        new_seq.x = np.array(self.x)
        return new_seq

    def append(self, value):
        self.x = np.concatenate([self.x, np.reshape(value, [self.get_level(), 1])], axis=1)

    def __len__(self):
        return self.get_length()

    def __str__(self):
        max_value = self.x.max()
        max_width = len(str(max_value))

        s_all = ''
        for i, x in enumerate(self.x):
            s = ' '.join(map(lambda n: '{{:>{}}}'.format(max_width).format(n), x))
            label = '[w0]' if i == 0 else '[h%d]' % i
            s_all += label + ' ' + s + '\n'

        return s_all

    def __add__(self, other):
        s = Seq()
        s.x = np.concatenate([self.x, other.x], axis=1)
        return s

    def __iadd__(self, other):
        self.x = np.concatenate([self.x, other.x], axis=1)
        return self

    def __getitem__(self, item):
        s = Seq()
        s.x = self.x[:, item]
        return s


def seq_list_get(seq_list, level):
    """
    Args:
        seq_list: a list of Seq()
        level:  level

    Returns:
        a list of list
    """
    return [s.x[level].tolist() for s in seq_list]


def get_x(seq_list):
    return seq_list_get(seq_list, 0)


def get_h(seq_list):
    return seq_list_get(seq_list, 1)


def seq_list_enumerate_tag(seq_list, tag_size, pos):

    if isinstance(pos, int) or isinstance(pos, np.int32) or isinstance(pos, np.int64):
        pos = [pos] * len(seq_list)

    res_list = []
    for s, i in zip(seq_list, pos):
        temps = s.copy()
        for tag in range(tag_size):
            temps.x[1, i] = tag
            res_list.append(temps.copy())
    return res_list


def list_create(x_list, tag_list):
    return [Seq([x, t]) for x, t in zip(x_list, tag_list)]


def list_repeat(seq_list, n):
    """
    input: [seq1, seq2], n=2
    output: [seq1, seq1, seq2, seq2]
    """
    res_list = []
    for s in seq_list:
        res_list += [s.copy() for _ in range(n)]
    return res_list


def list_copy(seq_list):
    """copy all the seqs in the list"""
    return [s.copy() for s in seq_list]


def list_print(seq_list):
    print('[seq_list]:')
    for i, s in enumerate(seq_list):
        print(s)


def hard_class_seqs(word_seq_list, word_to_class=None):
    """
    transform a word sequence to a Seq with hard-class
    Args:
        word_seq_list: a list of word list
        word_to_class: a np.array mapping word-id to class-id

    Returns:
        a class list
    """
    seq_list = []
    for word_seq in word_seq_list:
        seq = Seq(len(word_seq), 2)
        seq.x[0] = np.array(word_seq)
        if word_to_class is not None:
            seq.x[1] = np.array(word_to_class[word_seq])
        seq_list.append(seq)
    return seq_list


class EmptyFile(object):
    def __iter__(self):
        return self

    def __next__(self):
        return None


class Data(object):
    """
    Read the multiple files including the word/char and tag informations
    """
    def __init__(self, vocab_files=None, train_list=None, valid_list=None, test_list=None,
                 beg_token='<s>', end_token='</s>', unk_token='<unk>',
                 len_count=None, max_len=102):
        """
        Args:
            vocab_files: a list, [word_vocab, tag_vocab]
            train_list: a list of tuple, [(word_file1, tag_file1), (word_file2, tag_file2), ...]
            valid_list: a list of tuple, similar to train_list
            test_list:  a list of tuple, similar to train_list
            beg_token:  str, beg_token_str
            end_token:  str, ..
            unk_token:  str, ..
            len_count: the length count
            max_len: the max_len
        """
        self.train_file_list = train_list
        self.valid_file_list = valid_list
        self.test_file_list = test_list

        self.beg_token = beg_token
        self.end_token = end_token
        self.unk_token = unk_token

        self.max_len = max_len
        self.len_count = len_count if len_count is not None else self.load_len_count(train_list[0])
        self.len_count = self.len_count[0: max_len+1]
        self.len_count[0] = 0

        if self.beg_token:
            self.len_count.insert(0, 0)
            self.max_len += 1
        if self.end_token:
            self.len_count.insert(0, 0)
            self.max_len += 1

        if vocab_files is not None:
            # load vocabs
            self.vocabs = [vocab.Vocab().read(x) for x in vocab_files]
        else:
            files_all = list(self.train_file_list)
            if self.valid_file_list is not None:
                files_all += self.valid_file_list
            if self.test_file_list is not None:
                files_all += self.test_file_list

            self.load_vocab(files_all, beg_token, end_token, unk_token)
        self.vocabs[0].create_chars()

        # load files
        print('[%s.%s] Load data...' % (__name__, self.__class__.__name__))
        self.datas = []
        self.datas.append(self.load_file(self.train_file_list[0]))
        if self.valid_file_list is not None:
            self.datas.append(self.load_file(self.valid_file_list[0]))
        if self.test_file_list is not None:
            self.datas.append(self.load_file(self.test_file_list[0]))

        # for training_data
        self.train_iter_cur_file = 0
        self.train_iter_cur_line = 0
        self.train_iter_cur_epoch = 0
        self.train_batch_size = 1000
        self.is_shuffle = True

    def load_len_count(self, file_tuple):
        return wb.file_len_count(file_tuple[0])

    def load_vocab(self, file_tuple_list, beg_token, end_token, unk_token):
        self.vocabs = []
        for files in zip(*file_tuple_list):
            v = vocab.Vocab().generate_vocab(list(files),
                                             add_beg_token=beg_token,
                                             add_end_token=end_token,
                                             add_unk_token=unk_token)
            self.vocabs.append(v)

    def load_file(self, file_tuple):
        seq_list = []
        fp_list = [EmptyFile() if fname is None else open(fname, 'rt') for fname in file_tuple]
        for line_tuple in zip(*fp_list):

            ids_list = []
            for v, line in zip(self.vocabs, line_tuple):
                if line is not None:
                    ws = line.split()
                    if self.beg_token is not None:
                        ws.insert(0, self.beg_token)
                    if self.end_token is not None:
                        ws.append(self.end_token)
                    ids = v.words_to_ids(ws, self.unk_token)
                    ids_list.append(ids)
                else:
                    ids_list.append(None)

            ids_len = [len(ids) for ids in filter(lambda x: x is not None, ids_list)]
            if not np.all(np.array(ids_len) == ids_len[0]):
                raise TypeError('[%s.%s] the lengths for each level is not equal\n' %
                                (__name__, self.__class__.__name__) +
                                'lengths={}\n'.format(ids_len) +
                                'line_tuple={}'.format(line_tuple))

            seq = Seq(ids_len[0], len(file_tuple))
            for i, ids in enumerate(ids_list):
                if ids is not None:
                    seq.x[i] = np.array(ids)

            # remove the empty and sequeces longer than max_len
            if seq.get_length() > self.max_len or seq.get_length() <= 0:
                continue

            seq_list.append(seq)

        return seq_list

    def load_data(self, chr_file, level=0, is_nbest=False):
        """ read the chr file to ids"""
        data = []
        with open(chr_file, 'rt') as f:
            for line in f:
                a = line.lower().split()
                if is_nbest:
                    a = a[1:]
                    if len(a) == 0:
                        a.append(self.end_token)

                if self.beg_token:
                    a.insert(0, self.beg_token)
                if self.end_token:
                    a.append(self.end_token)

                ids = self.vocabs[level].words_to_ids(a, self.unk_token)
                data.append(ids)

        return data

    def write_file(self, seq_list, file_name):
        with open(file_name, 'wt') as f:
            for seq in seq_list:
                f.write(str(seq) + '\n')

    def write_text(self, seq_list, file_name):
        """
        Args:
            seq_list:
            file_name: str or tuple of str.
                    if str, then write to one single files;
                    if a tuple of str, then write each level to different files

        Returns:
            None
        """
        if isinstance(file_name, str):
            with open(file_name, 'wt') as f:
                for seq in seq_list:
                    for v, ids in zip(self.vocabs, seq.x):
                        words = v.ids_to_words(ids)
                        f.write(' '.join(words) + '\n')
        else:
            for i, fname in enumerate(file_name):
                with open(fname, 'wt') as f:
                    for seq in seq_list:
                        words = self.vocabs[i].ids_to_words(seq.x[i])
                        f.write(' '.join(words) + '\n')

    def set_iter_config(self, batch_size, is_shuffle=True):
        self.train_batch_size = batch_size
        self.is_shuffle = is_shuffle

        if self.is_shuffle:
            np.random.shuffle(self.datas[0])

    def __iter__(self):
        return self

    def __next__(self):
        data_seqs = self.datas[0][self.train_iter_cur_line: self.train_iter_cur_line + self.train_batch_size]
        self.train_iter_cur_line += self.train_batch_size

        if len(data_seqs) == self.train_batch_size:
            return data_seqs

        # read the next files
        self.train_iter_cur_file += 1
        self.train_iter_cur_line = 0
        if self.train_iter_cur_file >= len(self.train_file_list):
            # after one epoch
            self.train_iter_cur_epoch += 1
            self.train_iter_cur_file = 0
            if self.is_shuffle:
                np.random.shuffle(self.train_file_list)  # shuffle the files

        self.datas[0] = self.load_file(self.train_file_list[self.train_iter_cur_file])
        if self.is_shuffle:
            np.random.shuffle(self.datas[0])  # shuffle sequences

        # append the new seqs
        need_lines = self.train_batch_size - len(data_seqs)
        data_seqs += self.datas[0][self.train_iter_cur_line: self.train_iter_cur_line + need_lines]
        self.train_iter_cur_line += need_lines

        return data_seqs

    def get_cur_epoch(self):
        return self.train_iter_cur_epoch + \
               self.train_iter_cur_file / len(self.train_file_list) + \
               self.train_iter_cur_line / len(self.datas[0]) / len(self.train_file_list)

    def get_epoch_step_num(self):
        return len(self.train_file_list) * len(self.datas[0]) // self.train_batch_size

    def get_min_len(self):
        min_len = 1
        while self.len_count[min_len] == 0:
            min_len += 1
        return min_len

    def get_max_len(self):
        max_len = len(self.len_count)-1
        return max_len

    def get_pi_true(self):
        pi_true = np.array(self.len_count) / np.sum(self.len_count)

        # remove zero probs
        if np.min(pi_true) <= 1e-8:
            pi_true[self.get_min_len():] += 1e-5
            pi_true /= np.sum(pi_true)
        return pi_true

    def get_pi0(self, pi_true=None, add=None):
        if pi_true is None:
            pi_true = self.get_pi_true()

        min_len = self.get_min_len()
        pi0 = np.array(pi_true)
        idx = np.argmax(pi0[min_len+1:]) + min_len + 1

        if add is None:
            pi0[min_len:idx+1] = pi0[idx]
        else:
            pi0[min_len:idx+1] += add
        pi0[0: min_len] = 0
        pi0 /= np.sum(pi0)
        return pi0

    def get_vocab_size(self):
        return self.vocabs[0].get_size()

    def get_char_size(self):
        return self.vocabs[0].get_char_size()

    def get_tag_size(self):
        return self.vocabs[1].get_size()

    def get_beg_tokens(self):
        return [v.word_to_id[self.beg_token] for v in self.vocabs]

    def get_end_tokens(self):
        return [v.word_to_id[self.end_token] for v in self.vocabs]

    def create_data(self, level=0):
        data = reader.Data()
        data.beg_token_str = self.beg_token
        data.end_token_str = self.end_token
        data.unk_token_str = self.unk_token
        data.word_to_id = self.vocabs[level].word_to_id
        data.word_list = self.vocabs[level].words
        data.word_count = self.vocabs[level].count

        for d in self.datas:
            data.datas.append(seq_list_get(d, level))

        return data


class DataX(Data):
    def __init__(self, total_level=2, vocab_files=None, train_list=None, valid_list=None, test_list=None,
                 beg_token='<s>', end_token='</s>', unk_token='<unk>',
                 len_count=None, max_len=102):

        self.total_level = total_level
        super().__init__(vocab_files, train_list, valid_list, test_list,
                         beg_token, end_token, unk_token, len_count, max_len)

    def load_len_count(self, file_tuple):
        if isinstance(file_tuple, tuple):
            return super().load_len_count(file_tuple)

        return wb.file_len_count(file_tuple)

    def load_vocab(self, file_tuple_list, beg_token, end_token, unk_token):
        # file_tuple_list is a list of files
        if isinstance(file_tuple_list[0], str):
            self.vocabs = []
            for i in range(self.total_level):
                v = vocab.VocabX(self.total_level, i).generate_vocab(file_tuple_list,
                                                                     add_beg_token=beg_token,
                                                                     add_end_token=end_token,
                                                                     add_unk_token=unk_token)
                self.vocabs.append(v)
        else:
            super().load_vocab(file_tuple_list, beg_token, end_token, unk_token)

    def load_file(self, file_tuple):
        if isinstance(file_tuple, tuple):
            return super().load_file(file_tuple)

        seq_list = []
        with open(file_tuple, 'rt') as f:
            for line1 in f:
                line_list = [line1]
                for i in range(1, self.total_level):
                    line_list.append(f.readline())

                ids_list = []
                for v, line in zip(self.vocabs, line_list):
                    ws = line.split()
                    if self.beg_token is not None:
                        ws.insert(0, self.beg_token)
                    if self.end_token is not None:
                        ws.append(self.end_token)
                    ids = v.words_to_ids(ws, self.unk_token)
                    ids_list.append(ids)

                ids_len = [len(ids) for ids in ids_list]
                if not np.all(np.array(ids_len) == ids_len[0]):
                    raise TypeError('[%s.%s] the lengths for each level is not equal\n' %
                                    (__name__, self.__class__.__name__) +
                                    'lengths={}\n'.format(ids_len) +
                                    'line_tuple={}'.format(line_list))

                seq = Seq(np.array(ids_list))

                # remove the empty and sequeces longer than max_len
                if seq.get_length() > self.max_len or seq.get_length() <= 0:
                    continue

                seq_list.append(seq)

        return seq_list


def tag_error(correct_list, predict_list):
    correct_num = 0
    for a1, a2 in zip(correct_list, predict_list):
        correct_num += np.sum(np.array(a1[1:-1]) == np.array(a2[1:-1]))

    currect_sum = np.sum([len(x)-2 for x in correct_list])
    predict_sum = np.sum([len(x)-2 for x in predict_list])

    P = 100.0 * correct_num / predict_sum
    R = 100.0 * correct_num / currect_sum
    F = 2 * P * R / (P + R) if (P + R > 0) else 0
    return P, R, F


if __name__ == '__main__':
    x1 = Seq()
    x2 = x1.copy()

    x1.random(vocab_sizes=[1000, 10])
    x2.random(vocab_sizes=[100, 10])

    print(x1)
    print(x2)
    print(x1[2: -1])
    x1 += x2
    print(x1)

