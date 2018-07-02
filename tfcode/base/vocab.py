import os
import json
import numpy as np


class Vocab(object):
    def __init__(self):
        self.word_to_id = dict()
        self.count = list()
        self.words = list()

        self.to_lower = False

        # add character information
        self.chars = list()   # ['a', 'b', 'c', 'd', ...]
        self.char_to_id = dict()  # {'a': 0, 'b': 1, ...}
        self.word_to_chars = list() # [ ['a', 'b', 'c'], ... ]
        self.word_max_len = 0
        self.char_beg_id = 0
        self.char_end_id = 0

    def load_data(self, file_list):
        v_count = dict()
        total_line = 0
        total_word = 0
        for file in file_list:
            print('[%s.%s] generate_vocab: ' % (__name__, self.__class__.__name__), file)
            with open(file, 'rt') as f:
                for line in f:
                    # to lower
                    if self.to_lower:
                        line = line.lower()

                    for w in line.split():
                        v_count.setdefault(w, 0)
                        v_count[w] += 1
                        total_word += 1
                    total_line += 1
        return v_count, total_line, total_word

    def generate_vocab(self, file_list, cutoff=0, max_size=None,
                       add_beg_token='<s>', add_end_token='</s>', add_unk_token='<unk>',
                       to_lower=False):

        self.to_lower = to_lower
        v_count, total_line, total_word = self.load_data(file_list)

        if add_beg_token is not None:
            v_count[add_beg_token] = total_line
        if add_end_token is not None:
            v_count[add_end_token] = total_line
        if add_unk_token is not None:
            v_count[add_unk_token] = 1

        print('[%s.%s] vocab_size=' % (__name__, self.__class__.__name__), len(v_count))
        print('[%s.%s] total_line=' % (__name__, self.__class__.__name__), total_line)
        print('[%s.%s] total_word=' % (__name__, self.__class__.__name__), total_word)

        # cutoff
        v_list = []
        ignore_list = [add_beg_token, add_end_token, add_unk_token]
        for w, count in v_count.items():
            if w in ignore_list:
                continue
            if count > cutoff:
                v_list.append((w, count))

        # to handle the words with the same counts
        v_list = sorted(v_list, key=lambda x: x[0])   # sorted as the word
        v_list = sorted(v_list, key=lambda x: -x[1])   # sorted as the count
        ignore_dict = dict()
        for ignore_token in reversed(ignore_list):
            if ignore_token is not None and ignore_token not in ignore_dict:
                v_list.insert(0, (ignore_token, v_count[ignore_token]))
                ignore_dict[ignore_token] = 0
        print('[%s.%s] vocab_size(after_cutoff)=' % (__name__, self.__class__.__name__), len(v_list))

        if max_size is not None:
            print('[%s.%s] vocab max_size=()' % (__name__, self.__class__.__name__), max_size)
            unk_count = sum(x[1] for x in v_list[max_size:])
            v_list = v_list[0: max_size]

            # revise the unkcount
            if add_unk_token is not None:
                for i in range(len(v_list)):
                    if v_list[i][0] == add_unk_token:
                        v_list[i] = (add_unk_token, v_list[i][1] + unk_count)
                        break

        # create vocab
        self.count = list()
        self.words = list()
        self.word_to_id = dict()
        for i, (w, count) in enumerate(v_list):
            self.words.append(w)
            self.count.append(count)
            self.word_to_id[w] = i

        return self

    def write(self, fname):
        with open(fname, 'wt') as f:
            f.write('to_lower = %d\n' % int(self.to_lower))
            for i in range(len(self.words)):
                f.write('{}\t{}\t{}'.format(i, self.words[i], self.count[i]))
                if self.word_to_chars:
                    s = ' '.join('{}'.format(k) for k in self.word_to_chars[i])
                    f.write('\t{}'.format(s))
                f.write('\n')

        # write a extra char vocabulary
        if self.chars:
            with open(fname + '.chr', 'wt') as f:
                f.write('char_beg_id = %d\n' % self.char_beg_id)
                f.write('char_end_id = %d\n' % self.char_end_id)
                f.write('word_max_len = %d\n' % self.word_max_len)
                f.write('id \t char\n')
                for i in range(len(self.chars)):
                    f.write('{}\t{}\n'.format(i, self.chars[i]))

    def read(self, fname):
        self.words = list()
        self.count = list()
        self.word_to_id = dict()
        self.word_to_chars = list()
        with open(fname, 'rt') as f:
            self.to_lower = bool(int(f.readline().split()[-1]))

            for line in f:
                a = line.split()
                i = int(a[0])
                w = a[1]
                n = int(a[2])

                self.words.append(w)
                self.count.append(n)
                self.word_to_id[w] = i

                # read word_to_chars
                if len(a) > 3:
                    self.word_to_chars.append([int(k) for k in a[3:]])

        if self.word_to_chars:
            # read char vocab
            self.chars = list()
            self.char_to_id = dict()
            with open(fname + '.chr', 'rt') as f:
                self.char_beg_id = int(f.readline().split()[-1])
                self.char_end_id = int(f.readline().split()[-1])
                self.word_max_len = int(f.readline().split()[-1])
                f.readline()
                for line in f:
                    a = line.split()
                    i = int(a[0])
                    c = a[1]
                    self.chars.append(c)
                    self.char_to_id[c] = i

        return self

    def create_chars(self, add_char_beg='<s>', add_char_end='</s>'):
        if self.chars:
            return

        # process the word and split to chars
        c_dict = dict()
        for w in self.words:
            for c in list(w):
                c_dict.setdefault(c, 0)
        if add_char_beg is not None:
            c_dict.setdefault(add_char_beg)
        if add_char_end is not None:
            c_dict.setdefault(add_char_end)

        self.chars = sorted(c_dict.keys())
        self.char_to_id = dict([(c, i) for i, c in enumerate(self.chars)])
        self.char_beg_id = self.char_to_id[add_char_beg]
        self.char_end_id = self.char_to_id[add_char_end]

        self.word_to_chars = []
        for w in self.words:
            chr_ids = [self.char_to_id[c] for c in w]
            chr_ids.insert(0, self.char_beg_id)
            chr_ids.append(self.char_end_id)
            self.word_to_chars.append(chr_ids)

        self.word_max_len = max([len(x) for x in self.word_to_chars])

    def words_to_ids(self, word_list, unk_token='<unk>'):
        id_list = []
        for w in word_list:
            if self.to_lower:
                w = w.lower()

            if w in self.word_to_id:
                id_list.append(self.word_to_id[w])
            elif unk_token is not None and unk_token in self.word_to_id:
                id_list.append(self.word_to_id[unk_token])
            else:
                raise KeyError('[%s.%s] cannot find the word = %s' % (__name__, self.__class__.__name__, w))
        return id_list

    def ids_to_words(self, id_list):
        return [self.words[i] for i in id_list]

    def get_size(self):
        return len(self.words)

    def get_char_size(self):
        if not self.chars:
            raise TypeError('[Vocab] no char information!!')
        return len(self.chars)

    def __contains__(self, item):
        return item in self.word_to_id


class VocabX(Vocab):
    def __init__(self, total_level=2, read_level=0):
        super().__init__()

        self.total_level = total_level
        self.read_level = read_level

    def load_data(self, file_list):
        v_count = dict()
        total_line = 0
        total_word = 0
        for file in file_list:
            print('[%s.%s] generate_vocab: ' % (__name__, self.__class__.__name__), file)
            cur_line = 0
            with open(file, 'rt') as f:
                for line in f:
                    if cur_line % self.total_level == self.read_level:
                        for w in line.split():
                            v_count.setdefault(w, 0)
                            v_count[w] += 1
                            total_word += 1
                        total_line += 1
                    cur_line += 1
        return v_count, total_line, total_word
