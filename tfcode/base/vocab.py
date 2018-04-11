import os
import numpy as np


class Vocab(object):
    def __init__(self):
        self.word_to_id = dict()
        self.count = list()
        self.words = list()

    def load_data(self, file_list):
        v_count = dict()
        total_line = 0
        total_word = 0
        for file in file_list:
            print('[%s.%s] generate_vocab: ' % (__name__, self.__class__.__name__), file)
            with open(file, 'rt') as f:
                for line in f:
                    for w in line.split():
                        v_count.setdefault(w, 0)
                        v_count[w] += 1
                        total_word += 1
                    total_line += 1
        return v_count, total_line, total_word

    def generate_vocab(self, file_list, cutoff=0, max_size=None, add_beg_token='<s>', add_end_token='</s>', add_unk_token='<unk>'):
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
            for i in range(len(self.words)):
                f.write('{}\t{}\t{}\n'.format(i, self.words[i], self.count[i]))

    def read(self, fname):
        self.words = list()
        self.count = list()
        self.word_to_id = dict()
        with open(fname, 'rt') as f:
            for line in f:
                a = line.split()
                i = int(a[0])
                w = a[1]
                if len(a) >= 3:
                    n = int(a[2])
                else:
                    n = 1

                self.words.append(w)
                self.count.append(n)
                self.word_to_id[w] = i

        return self

    def words_to_ids(self, word_list, unk_token='<unk>'):
        id_list = []
        for w in word_list:
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
