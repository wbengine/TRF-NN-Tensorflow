import json
import numpy as np
import time

from base import *


def write_value(f, value, name='value'):
    if isinstance(value, np.int64) or isinstance(value, np.int32):
        value = int(value)
    if isinstance(value, np.float32) or isinstance(value, np.float64):
        value = float(value)
    f.write('{} = {}\n'.format(name, json.dumps(value)))


def read_value(f):
    s = f.__next__()
    idx = s.find('= ')
    return json.loads(s[idx+1:])


def read_feattype_file(fname):
    type_dict = dict()
    with open(fname, 'rt') as f:
        for line in f:
            commemt = line.find('//')
            if commemt != -1:
                line = line[:commemt]
            a = line.split()
            if len(a) == 0:
                continue

            type = a[0]
            cutoff = 0
            if len(a) >= 2:
                s = a[1]
                if len(s) == 1:
                    cutoff = int(s)
                else:
                    cutoff = []
                    for c in s:
                        cutoff.append(int(c))
            type_dict[type] = cutoff
    return type_dict


def separate_type(type_dict):
    """return the type for word and class separately"""
    wt_dict = dict()
    ct_dict = dict()
    for key, v in type_dict.items():
        contain_word = key.find('w') != -1
        contain_class = key.find('c') != -1
        if contain_word and not contain_class:
            wt_dict[key] = v
        if contain_class and not contain_word:
            ct_dict[key] = v

    return wt_dict, ct_dict


def trie_write(f, input_trie, input_weight=None, sorted=False):
    write_num = 0
    for key, id in trie.TrieIter(input_trie, sorted):
        if id is None:
            continue

        if input_weight is not None:
            v = input_weight[id]
        else:
            v = 0
        f.write('key={} id={} value={}\n'.format(json.dumps(key), id, v))
        write_num += 1
    return write_num


def trie_read(f, num, input_trie, input_weight=None):
    for i in range(num):
        s = f.__next__()
        s = s.replace('key=', '|')
        s = s.replace(' id=', '|')
        s = s.replace(' value=', '|')
        a = list(filter(None, s.split('|')))
        key = json.loads(a[0])
        id = int(a[1])
        v = float(a[2])
        input_trie.insert(key, id)
        if input_weight is not None:
            input_weight[id] = v


def seq_to_ngrams(seq, order):
    ngrams = []
    for i in range(0, len(seq)-order+1):
        ngrams.append(seq[i: i+order])
    return ngrams


class feat:
    def __init__(self, type_dict=None, opt_config={}):
        """
        create several type of features
        :param type_dict: the type dict, key=type name, value=cutoff list or integer, such as
            {'w[1:4]': [0, 0, 0, 2], 'w[1]-[1]w[2]', 2}
        :param opt_conifg: dict, such as
            {'name': 'adam', 'max_norm': 5}
            {'name': 'sgd', 'max_norm': 5}
            {'name': 'var', 'vgap': 1e-6}  # using empirical variance to rescale the gradient
        """
        self.opt_config = dict(opt_config)
        if type_dict is None:
            return
        self.feat_list = []
        self.type_dict = type_dict
        for key, v in type_dict.items():
            self.feat_list.append(feat_onetype(key))
        self.num = 0
        self.max_order = max([f.max_order for f in self.feat_list])
        self.create_values_buf(0)

        self.is_precompute_train_expec = False
        self.need_be_rescaled_by_var = False   # denote the gradient will be rescaled by empirical variance

        self.find_time = 0

    def create_values_buf(self, feat_num):
        self.num = feat_num
        self.values = np.zeros(feat_num)
        self.train_expec = np.zeros(feat_num)
        self.sample_expec = np.zeros(feat_num)

        self.empirical_var = None  # used to store the empirical variance to rescale the gradient

        # update method
        if self.opt_config['name'].lower() == 'adam':
            self.opt = wb.ArrayUpdate(self.values, self.opt_config)
        elif self.opt_config['name'].lower() == 'sgd':
            self.opt = wb.ArrayUpdate(self.values, self.opt_config)
        elif self.opt_config['name'].lower() == 'var':
            self.opt = wb.ArrayUpdate(self.values, {'name': 'sgd'})  # using the gradient descent
            self.need_be_rescaled_by_var = True
            self.empirical_var = np.zeros(feat_num)
            self.empirical_var_gap = self.opt_config.setdefault('vgap', 1e-12)
        else:
            raise TypeError("Unknown opt name: " + self.opt_config['name'])

        self.is_precompute_train_expec = False

    def load_from_seqs(self, seqs):
        self.num = 0
        for ftype in self.feat_list:
            for seq in seqs:
                self.num = ftype.exact(seq, self.num)

        self.create_values_buf(self.num)
        print('Load features {:,}'.format(self.num))

    def write(self, f, value_buf=None):
        if value_buf is None:
            value_buf = self.values

        write_value(f, self.type_dict, 'feature_type')
        write_value(f, self.num, 'feature_total_num')
        write_value(f, self.max_order, 'feature_max_order')
        for ftype in self.feat_list:
            ftype.write(f, value_buf)

    def read(self, f):
        type_dict = read_value(f)
        self.__init__(type_dict, self.opt_config)

        self.num = read_value(f)
        self.max_order = read_value(f)
        self.create_values_buf(self.num)
        for ftype in self.feat_list:
            ftype.read(f, self.values)

    def read_nocreate(self, f, value_buf):
        type_dict = read_value(f)
        num = read_value(f)
        max_order = read_value(f)

        # make sure this file match current Feat instance
        if type_dict != self.type_dict or \
           num != self.num or \
           max_order != self.max_order:
            print('[E] this file is not match current Feat instance !')
            print('[E] self.type_dict={}'.format(self.type_dict))
            print('[E] self.num={}'.format(self.num))
            print('[E] self.max_order={}'.format(self.max_order))
            print('[E] curr.type_dict={}'.format(type_dict))
            print('[E] curr.num={}'.format(num))
            print('[E] curr.max_order={}'.format(max_order))
            raise Exception('Instance and file do not Match')

        assert value_buf is not None
        for ftype in self.feat_list:
            ftype.read(f, value_buf)

    def write_train_expec(self, fname):
        with open(fname, 'wt') as f:
            print('[Feat] write empirical expectation to %s' % fname)
            self.write(f, self.train_expec)

    def read_train_expec(self, fname):
        with open(fname, 'rt') as f:
            print('[Feat] read empirical expectation from %s' % fname)
            self.read_nocreate(f, self.train_expec)

    def write_empirical_var(self, fname):
        with open(fname, 'wt') as f:
            print('[Feat] write empirical variance to %s' % fname)
            self.write(f, self.empirical_var)

    def read_empirical_var(self, fname):
        with open(fname, 'rt') as f:
            print('[Feat] read empirical variance from %s' % fname)
            self.read_nocreate(f, self.empirical_var)

    def ngram_find(self, ngrams):
        """ return a list containing the existing feature id """
        res = []
        for ngram in ngrams:
            a = []
            for ftype in self.feat_list:
                a += ftype.ngram_find([ngram])[0]
            res.append(a)
        return res

    def ngram_weight(self, ngrams):
        """
        find the feature observed in ngrams
        :param ngrams: 2D array, of size (batch_size, order), the order should be <= feat.max_order
        :return: list of list, containing all the feature id
        """
        # o = np.zeros(len(ngrams))
        # for ftype in self.feat_list:
        #     o += ftype.ngram_weight(ngrams, self.values)
        # return o

        ids = self.ngram_find(ngrams)
        return np.array([np.sum(self.values[x]) for x in ids])

    def seq_find(self, seq):
        """
        find the features observed in given sequence
        Args:
            seq: a list

        Returns:

        """
        tbeg = time.time()
        a = []
        for ftype in self.feat_list:
            a += ftype.seq_find(seq)
        self.find_time += (time.time() - tbeg) / 60
        return a

    def seq_weight(self, seq):
        """
        Return the summation of weight of observed features
        Args:
            seq: a list

        Returns:

        """
        return np.sum(self.values[self.seq_find(seq)])

    def seq_list_weight(self, seq_list):
        """
        Return the weight of a list of sequences
        Args:
            seq_list:

        Returns:

        """
        return np.array([self.seq_weight(seq) for seq in seq_list])

    def ngram_count(self, expec, ngrams, scalar):
        find_feats = self.ngram_find(ngrams)
        for ids, d in zip(find_feats, scalar):
            for i in ids:
                expec[i] += d

    def precompute_train_expec(self, train_ngrams, train_scalar):
        """trans_scalar should be a double"""
        self.ngram_count(self.train_expec, train_ngrams, np.ones(len(train_ngrams)) * train_scalar)
        self.is_precompute_train_expec = True

    def update(self, train_ngrams, train_scalar, sample_ngrams, sample_scalar, lr=0.001, L2_reg=0.):
        if not self.is_precompute_train_expec:
            self.train_expec.fill(0)
            self.ngram_count(self.train_expec, train_ngrams, train_scalar)

        self.sample_expec.fill(0)
        self.ngram_count(self.sample_expec, sample_ngrams, sample_scalar)
        grads = self.sample_expec - self.train_expec + L2_reg * self.values

        # update
        self.values += self.opt.update(grads, lr)
        return self.opt.dx_norm

    def seq_count(self, expec, seq_list, scalar):
        for seq, d in zip(seq_list, scalar):
            for i in self.seq_find(seq):
                expec[i] += d

    def precompute_seq_train_expec(self, seq_list, scalar, logname=None):
        """
        compute the empirical expectation on the training set.
        Args:
            seq_list: a list of sequences
            scalar: scalar of each sequence
            logname: read the existing value form log files

        Returns:
            None
        """

        # if exist, then read the values
        if logname is not None and wb.exists(logname + '.mean'):
            self.read_train_expec(logname + '.mean')
        else:
            print('[Feat] count the empirical expectation ...')
            self.seq_count(self.train_expec, seq_list, np.ones(len(seq_list)) * scalar)
            self.is_precompute_train_expec = True
            if logname is not None:
                self.write_train_expec(logname + '.mean')

        if self.need_be_rescaled_by_var:
            if logname is not None and wb.exists(logname + '.var'):
                self.read_empirical_var(logname + '.var')
            else:
                # compute the empirical variance
                print('[feat] compute the empircal vairance ...')
                self.empirical_var.fill(0)
                seq_n = [len(x) for x in seq_list]
                min_n = min(seq_n)
                max_n = max(seq_n)
                for l in range(min_n, max_n+1):
                    pf = np.zeros(self.num)
                    pf2 = np.zeros(self.num)

                    idx = np.where(np.array(seq_n) == l)[0]
                    if len(idx) == 0:
                        print('[Feat] [W] number of length {} is ZERO'.format(l))
                        continue

                    for i in idx:
                        seq = seq_list[i]
                        find_dict = dict()
                        for feat_n in self.seq_find(seq):
                            find_dict.setdefault(feat_n, 0)
                            find_dict[feat_n] += 1
                        for feat_n, feat_count in find_dict.items():
                            pf[feat_n] += feat_count
                            pf2[feat_n] += feat_count ** 2
                    pf /= len(idx)
                    pf2 /= len(idx)
                    self.empirical_var += (pf2 - pf ** 2) * len(idx) / len(seq_list)
                if logname is not None:
                    self.write_empirical_var(logname + '.var')

    def seq_update(self, train_seq_list, train_scalar, sample_seq_list, sample_scalar, lr, L2_reg=0, dropout=None):
        """
        given the training sequence and sample sequence, update the parameters
        Args:
            train_seq_list:
            train_scalar:
            sample_seq_list:
            sample_scalar:
            lr:
            L2_reg:
            dropout:

        Returns:

        """
        if not self.is_precompute_train_expec:
            self.train_expec.fill(0)
            self.seq_count(self.train_expec, train_seq_list, train_scalar)

        self.sample_expec.fill(0)
        self.seq_count(self.sample_expec, sample_seq_list, sample_scalar)
        grads = self.sample_expec - self.train_expec + L2_reg * self.values

        if self.need_be_rescaled_by_var:
            # rescaled by empirical variance
            grads /= np.maximum(self.empirical_var_gap, self.empirical_var + L2_reg)

        if dropout is not None:
            assert dropout >= 0
            grads *= np.random.binomial(n=1, p=1-dropout, size=grads.shape)

        self.values += self.opt.update(grads, lr)
        return self.opt.dx_norm


class feat_onetype:
    def __init__(self, type=''):
        """
        create a set of features
        :param type: a string denoting the feature type, such as "w[1:4]" or "w[1]-[1]w[1]"
        """
        self.type = type
        if type != '':
            self.map_list = self.analyze_type(type)
            self.max_order = max([-np.min(m)+1 for m in self.map_list])
        self.trie = trie.trie()
        self.num = 0

    def analyze_type(self, type):
        idx = type.find(':')
        if idx == -1:
            type_list = [type]
        else:
            beg = type.rfind('[', 0, idx)
            end = type.find(']', idx)
            v1 = int(type[beg+1: idx])
            v2 = int(type[idx+1: end])
            fmt = type[0:beg+1] + '{}' + type[end:]
            type_list = [fmt.format(i) for i in range(v1, v2+1)]

        map_list = []
        for t in type_list:
            a = filter(None, t.split(']'))
            n = 0
            p = []
            for s in a:
                i = int(s[2:])
                if s[0] != '-':
                    p += list(range(n, n+i))
                n += i
            p = np.array(p) - n + 1
            map_list.append(p)
        return map_list

    def exact_key(self, seq):
        """exact all the keys observed in the given sequence, used to find features"""
        key_list = []
        seq = np.array(seq)
        for m in self.map_list:
            n = -np.min(m) + 1
            if n == 1:
                # for unigram, skip the unigram for begin-token and end-token
                seq_revise = seq[1:-1]
                for i in range(0, len(seq_revise)-n+1):
                    key = seq_revise[i:i+n][m+n-1]
                    key_list.append(key.tolist())
            else:
                for i in range(0, len(seq)-n+1):
                    key = seq[i:i+n][m+n-1]
                    key_list.append(key.tolist())
        return key_list

    def exact(self, seq, beg_id=0):
        for key in self.exact_key(seq):
            sub = self.trie.setdefault(key, beg_id)
            if sub.data == beg_id:  # add successfully
                beg_id += 1
                self.num += 1
        return beg_id

    def ngram_find(self, ngrams):
        """
        find the feature observed in ngrams
        :param ngrams: 2D array, of size (batch_size, order), the order should be <= feat.max_order
        :return: list of list, containing all the feature id
        """
        res = []
        for ngram in ngrams:
            n = len(ngram)
            ids = []
            for m in self.map_list:
                key = list(ngram[m+n-1])
                id = self.trie.find(key)
                if id is not None:
                    ids.append(id)
            res.append(ids)
        return res

    def ngram_weight(self, ngrams, values):
        ids = self.ngram_find(ngrams)
        return [np.sum(values[np.array(x, dtype='int64')]) for x in ids]

    def seq_find(self, seq):
        """input a sequence, and find the observed features"""
        ids = []
        for key in self.exact_key(seq):
            id = self.trie.find(key)
            if id is not None:
                ids.append(id)
        return ids

    def write(self, f, values=None):
        f.write('feat_type = {}\n'.format(self.type))
        f.write('feat_num = {}\n'.format(self.num))

        write_num = 0
        for key, id in trie.TrieIter(self.trie):
            if id is None:
                continue

            if values is not None:
                v = values[id]
            else:
                v = 0
            f.write('key={} id={} value={}\n'.format(json.dumps(key), id, v))
            write_num += 1

        assert write_num == self.num

    def read(self, f, values=None):
        self.type = f.__next__().split()[-1]
        self.__init__(self.type)

        self.num = int(f.__next__().split()[-1])
        for i in range(self.num):
            s = f.__next__()
            s = s.replace('key=', '|')
            s = s.replace(' id=', '|')
            s = s.replace(' value=', '|')
            a = list(filter(None, s.split('|')))
            key = json.loads(a[0])
            id = int(a[1])
            v = float(a[2])
            self.trie.insert(key, id)
            if values is not None:
                values[id] = v


def demo():

    data = reader.Data().load_raw_data(reader.word_raw_dir())

    type(data.datas[0][0])

    wt, ct = separate_type(read_feattype_file('g2.fs'))

    print('local features')
    f = feat(wt)
    f.load_from_seqs(data.datas[0])
    print('num=', f.num)

    wgrams = np.array([[1, 2, 3, 4], [0, 0, 1, 2]])
    print('wgram=', wgrams)

    print(f.ngram_find(wgrams))
    print(f.ngram_weight(wgrams))

    with open('test_feat.txt', 'wt') as file:
        f.write(file)

if __name__ == '__main__':
    demo()



