import json
import os

from base import *
from multiprocessing import Process, Manager, Queue, Value


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


class Feats(object):
    def __init__(self, type_dict, map_tag_to_level={'w': 0, 'c': 1}):
        """
        create a collection of types of features
        Args:
            type_dict: the type dict, key=type name, value=cutoff list or integer, such as
                    {
                        'w[1:4]': [0, 0, 0, 2],
                        'w[1]-[1]w[2]':2
                    }
            map_tag_to_level: a dict
        """
        self.type_dict = type_dict
        self.map_tag_to_level = map_tag_to_level

        self.feat_list = []
        for key, v in type_dict.items():
            self.feat_list.append(SingleFeat(key, v, map_tag_to_level))
        self.num = 0
        self.values = []
        self.create_values_buf(0)

    def get_order(self, level=None):
        return max([f.get_order(level) for f in self.feat_list])

    def create_values_buf(self, feat_num):
        self.num = feat_num
        self.values = np.zeros(feat_num)

    def load_from_seqs(self, seqs):
        self.num = 0
        for ftype in self.feat_list:
            self.num = ftype.create_from_seqs(seqs, self.num)

        self.create_values_buf(self.num)
        # print('[{}.{}] Load features {:,}'.format(__name__, self.__class__.__name__, self.num))

    def insert_single_feat(self, single_feat):
        """
        Insert a single feat into the package
        Args:
            single_feat: SingleFeat

        Returns:

        """
        self.feat_list.append(single_feat)
        self.num += single_feat.num
        print('[%s.%s] recreate value buf = %d.' % (__name__, self.__class__.__name__, self.num))
        self.create_values_buf(self.num)

    def save(self, f, value_buf=None):
        if value_buf is None:
            value_buf = self.values

        write_value(f, self.type_dict, 'feature_type')
        write_value(f, self.map_tag_to_level, 'map_tag_to_level')
        write_value(f, self.num, 'feature_total_num')
        write_value(f, self.get_order(), 'feature_max_order')
        for ftype in self.feat_list:
            ftype.write(f, value_buf)

    def restore(self, f):
        type_dict = read_value(f)
        map_tag = read_value(f)
        self.__init__(type_dict, map_tag)

        self.num = read_value(f)
        read_value(f)
        self.create_values_buf(self.num)
        for ftype in self.feat_list:
            ftype.read(f, self.values)

    @staticmethod
    def load(fp):
        type_dict = read_value(fp)
        map_tag = read_value(fp)
        f = Feats(type_dict, map_tag)

        f.num = read_value(fp)
        f.max_order = read_value(fp)
        f.create_values_buf(f.num)
        for ftype in f.feat_list:
            ftype.read(fp, f.values)
        return f

    def seq_find(self, seq, depend_on=None):
        a = []
        for ftype in self.feat_list:
            a += ftype.seq_find_depend(seq, depend_on)
        return a

    def seq_weight(self, seq, depend_on=None):
        return np.sum(self.values[self.seq_find(seq, depend_on)])

    def ngram_find(self, seq):
        return sum([ftype.ngram_find(seq) for ftype in self.feat_list], [])

    def ngram_weight(self, seq):
        return np.sum(self.values[self.ngram_find(seq)])

    def level_ngram_find(self, seq, level, pos, order):
        return sum([ftype.level_ngram_find(seq, level, pos, order) for ftype in self.feat_list], [])

    def level_ngram_weight(self, seq, level, pos, order):
        return np.sum(self.values[self.level_ngram_find(seq, level, pos, order)])

    def seq_list_find(self, seq_list, depend_on=None):
        return [self.seq_find(seq, depend_on) for seq in seq_list]

    def seq_list_weight(self, seq_list, depend_on=None):
        return np.array([self.seq_weight(seq, depend_on) for seq in seq_list])

    def seq_list_count(self, seq_list, seq_scalar):
        """
        compute sum_x d * f(x)
        Args:
            seq_list: a list of Seq()
            seq_scalar: a list of float

        Returns:
            a array of size self.num
        """
        buf = np.zeros_like(self.values)
        a_list = self.seq_list_find(seq_list)
        for a, d in zip(a_list, seq_scalar):
            for i in a:
                buf[i] += d
        return buf

    def seq_list_count2(self, seq_list, seq_scalar):
        """
        compute Sum_x d * f(x)^2
        Args:
            seq_list: a list of Seq()
            seq_scalar: a list of float

        Returns:
            a array of size self.num
        """
        buf = np.zeros_like(self.values)
        a_list = self.seq_list_find(seq_list)
        for a, d in zip(a_list, seq_scalar):
            f_count = dict()
            for i in a:
                f_count.setdefault(i, 0)
                f_count[i] += 1
            for i, n in f_count.items():
                buf[i] += d * n * n
        return buf


class FastFeats(Feats):
    def __init__(self, type_dict, map_tag_to_level={'w': 0, 'c': 1}, sub_process_num=4):
        super().__init__(type_dict, map_tag_to_level)

        self.sub_process_num = sub_process_num
        self.task_queue = Queue(maxsize=10)
        self.res_queue = Queue(maxsize=10)
        self.sub_processes = [Process(target=self.sub_process, args=(self.task_queue, self.res_queue))
                              for _ in range(self.sub_process_num)]

    def __del__(self):
        if self.sub_processes[0].is_alive():
            self.release()

    def start(self):
        for p in self.sub_processes:
            p.start()

    def release(self):
        for _ in range(self.sub_process_num):
            self.task_queue.put((-1, None, None))

        for p in self.sub_processes:
            p.join()

    def sub_process(self, task_queue, res_queue):
        print('[FastFeat] sub-process %d, start' % os.getpid())
        while True:
            tsk = task_queue.get()  # tuple( id, seq_list, depend_on )
            if tsk[0] == -1:
                break

            a_list = [self.seq_find(seq, tsk[2]) for seq in tsk[1]]
            res_queue.put((tsk[0], a_list))  # tuple( id, list )

        print('[FastFeat] sub-process %d, finished' % os.getpid())

    def seq_list_find(self, seq_list, depend_on=None):
        if not self.sub_processes[0].is_alive():
            self.start()

        # add task
        batch_size = int(np.ceil(len(seq_list) / self.sub_process_num))
        tsk_num = 0
        for batch_beg in range(0, len(seq_list), batch_size):
            self.task_queue.put((tsk_num, seq_list[batch_beg: batch_beg + batch_size], depend_on))
            tsk_num += 1

        # collect results
        res_dict = {}
        for _ in range(tsk_num):
            i, x = self.res_queue.get()
            res_dict[i] = x

        res = []
        for i in range(tsk_num):
            res += res_dict[i]

        assert len(res) == len(seq_list)
        return res

    def seq_list_weight(self, seq_list, depend_on=None):
        a_list = self.seq_list_find(seq_list, depend_on)
        return np.array([np.sum(self.values[a]) for a in a_list])


class KeyMap(object):
    """
    a key map used to map the input sequence or ngram to the key of trie
    """
    def __init__(self, type, map_tag_to_level):
        """
        Args:
            type: Single Type, such as w[1], w[2], w[1]c[1] as so on
        """
        self.map_tag_to_level = map_tag_to_level
        self.level_map, self.pos_map = self.parse(type)

    def get_order(self, level=None):
        """return the order for each level"""
        if level is None:
            return self.pos_map[-1] + 1

        idx = np.where([i == level for i in self.level_map])[0]
        if len(idx) == 0:
            return 0
        cur_pos = self.pos_map[idx]
        return cur_pos[-1] - cur_pos[0] + 1

    def parse(self, type):
        """
        Get the detail of the KeyMap, such as:
            if w->0, c->1
            w[1] -> level_map=[0], pos_map=[0]
            w[2] -> level_map=[0, 0], pos_map=[0, 1]
            w[1]c[1]w[1] -> level_map=[0, 1, 0], pos_map=[0, 1, 2]
            wc[2] -> level_map=[0, 0, 1, 1], pos_map=[0, 1, 0, 1]
        Args:
            type: str

        Returns:
            level_map, pos_map
        """
        if type.find(':') != -1:
            raise TypeError('[%s.%s] only support signle type!' % (__name__, self.__class__.__name__))

        a = filter(None, type.split(']'))
        n = 0
        levels = []
        poss = []
        for s in a:
            tags, nums = s.split('[')
            i = int(nums)
            pos = list(range(n, n+i))

            for tag in tags:
                if tag in self.map_tag_to_level:
                    levels += [self.map_tag_to_level[tag]] * len(pos)
                    poss += pos

            n += i

        # poss = np.array(poss)
        # levels = np.array(levels)

        return levels, poss

    def level_ngram_key(self, seq, level, pos, order):
        """get features for conditional forward-backward algorithms"""

        if self.get_order(level) > order:
            return None

        idx = np.where([i == level for i in self.level_map])[0]
        if len(idx) == 0:
            return None

        level_pos = [self.pos_map[i] for i in idx]
        offset = -level_pos[0]

        cur_pos = [i + pos + offset for i in self.pos_map]

        if np.min(cur_pos) < 0:
            return None
        if np.max(cur_pos) > len(seq) - 1:
            return None

        key = seq.x[self.level_map, cur_pos]
        return key

    def ngram_key(self, ngram):
        """
        input a ngram (Seq()) get the keys. For example:
            ngram:  [[x, y, z] [1,2,3]]
            return for different type
                w[2]:      [x, y]
                w[1]c[1]:  [x, 2]
                wc[2]:     [x, y, 1, 2]
                c[1]w[1]:  [1, y]
        Args:
            ngram: a Seq() denoting the ngrams

        Returns:
            a key (list)
        """
        if len(ngram) < self.get_order():
            return None

        key = ngram.x[self.level_map, self.pos_map]
        return key

    def get_keys(self, seq, depend_on=None):
        """
        get the key in seq depending on speciafied level and position
        Args:
            seq: a Seq()
            depend_on: two cases:
                1. (int > 0, int >0), indicating the (level, position).
                    find the features depending on given level and given position;
                2. int, indicating the level.
                    find all the features only depending on the given level

        Returns:
            a list of key (list)
        """

        if max(self.level_map) >= seq.get_level():
            raise TypeError('[%s.%s] the need level larger than the sequence level\n' %
                            (__name__, self.__class__.__name__) +
                            'level_map.max={} seq.get_level={}'.format(max(self.level_map), seq.get_level()))

        key_list = []
        order = self.pos_map[-1] + 1  # order

        if depend_on is None:
            for pos in range(0, seq.get_length()-order+1):
                key = [seq.x[i, j + pos] for i, j in zip(self.level_map, self.pos_map)]
                # seq.x[self.level_map, self.pos_map + pos]
                key_list.append(key)
        elif isinstance(depend_on, int):
            # find all the features independing on the given level
            if np.any([i != depend_on for i in self.level_map]):
                return []
            for pos in range(0, seq.get_length()-order+1):
                key = [seq.x[i, j + pos] for i, j in zip(self.level_map, self.pos_map)]
                key_list.append(key)
        else:
            depend_on_level, depend_on_pos = depend_on

            if depend_on_pos is not None:
                pos_range = range(max(0, depend_on_pos-order+1), min(depend_on_pos+1, seq.get_length()-order+1))
            else:
                pos_range = range(0, len(seq)-order+1)

            def equal(level, pos):
                if depend_on[0] is not None and depend_on[0] != level:
                    return False
                if depend_on[1] is not None and depend_on[1] != pos:
                    return False
                return True

            for pos in pos_range:
                cur_pos = [i + pos for i in self.pos_map]

                is_equal = False
                for l, p in zip(self.level_map, cur_pos):
                    if equal(l, p):
                        is_equal = True
                        break
                if is_equal:
                    key = [seq.x[i, j] for i, j in zip(self.level_map, cur_pos)]
                    key_list.append(key)

        return key_list


class SingleFeat(object):
    def __init__(self, type='', cut_off=None, map_tag_to_level={'w': 0, 'c': 1}):
        """
        Create a set of features
        Args:
            type: such as "w[1:4]" or "w[1]-[1]w[1]"
            map_tag_to_level: map tag('w', 'c') to level index in Seq()
        """
        self.type = type
        self.cut_off = cut_off
        self.map_tag_to_level = map_tag_to_level
        if type != '':
            self.map_list, self.cut_list = self.analyze_type(type)
            # self.max_order = max([m.get_order() for m in self.map_list])
        self.trie = trie.trie()
        self.num = 0

    def get_order(self, level=None):
        return max([m.get_order(level) for m in self.map_list])

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
        cut_list = []
        for i, t in enumerate(type_list):
            map_list.append(KeyMap(t, self.map_tag_to_level))
            if self.cut_off is None:
                cut_list.append(0)
            elif isinstance(self.cut_off, list):
                cut_list.append(self.cut_off[i])
            else:
                cut_list.append(self.cut_off)
        return map_list, cut_list

    def exact_key(self, seq):
        """
        exact all the keys observed in the given sequence, used to find features
        Args:
            seq: a Seq()

        Returns:
            a list of key
        """
        key_list = []
        for m in self.map_list:
            key_list += m.get_keys(seq)
        return key_list

    def exact(self, seq, beg_id=0):
        for key in self.exact_key(seq):
            sub = self.trie.setdefault(key, beg_id)
            if sub.data == beg_id:  # add successfully
                beg_id += 1
                self.num += 1
        return beg_id

    def create_from_seqs(self, seq_list, beg_id=0):
        """
        create the features from the sequences
        Args:
            seq_list: a list of Seq()
            beg_id: the beginning id of features

        Returns:
            None
        """
        count_trie = trie.trie()  # used to save the count

        for key_map, cut_num in zip(self.map_list, self.cut_list):  # for each KeyMap
            for seq in seq_list:
                if key_map.get_order() == 1:
                    # skip the beg/end token unigram
                    keys = key_map.get_keys(seq.get_sub(1, -1))
                else:
                    keys = key_map.get_keys(seq)
                for k in keys:
                    count = count_trie.setdefault(k, 0)
                    count.data += 1
                    if count.data >= cut_num:
                        # add the feature
                        id = self.trie.setdefault(k, beg_id)
                        if id.data == beg_id:
                            beg_id += 1
                            self.num += 1
        return beg_id

    def add_ngram(self, ngram_list, beg_id=0):
        """
        add a list of ngrams directly into the features
        Args:
            ngram_list: list of ngrams
            beg_id: the begin id

        Returns:
            the final id
        """
        # f = open('temp.txt', 'wt')
        for key in ngram_list:
            if not key:
                raise TypeError('[%s.%s] input an empty ngram key!' % (__name__, self.__class__.__name__))
            sub = self.trie.setdefault(key, beg_id)
            # f.write('{}  {}\n'.format(str(key), sub.data))
            if sub.data == beg_id:  # add successfully
                beg_id += 1
                self.num += 1
        # f.close()
        return beg_id

    def seq_find(self, seq, restrict_level=None):
        """
        input a Seq(), find all the observed features satisfying the constrains.
        Args:
            seq: a Seq()
            restrict_level: integer/str/None, if not None, find the features only depending on the given level, such as:
                restrict_level=0 or 'w', find all the word features, w[1:n], w[1]-[1]w[1]
                                        No c[1:n], w[1]-[1]c[1]

        Returns:
            a list of integer indicating the id of features
        """
        if restrict_level is not None:
            if isinstance(restrict_level, str):
                restrict_level = self.map_tag_to_level[restrict_level]
            for m in self.map_list:
                # if some map needs the level different with restrict_level
                # then return empty.
                if np.any(m.level_map != restrict_level):
                    return []

        ids = []
        for key in self.exact_key(seq):
            id = self.trie.find(key)
            if id is not None:
                ids.append(id)
        return ids

    def seq_find_depend(self, seq, depend_on=None):
        ids = []
        for m in self.map_list:
            for key in m.get_keys(seq, depend_on):
                id = self.trie.find(key)
                if id is not None:
                    ids.append(id)
        return ids

    def ngram_find(self, ngram):
        """
        input a ngram (a sub-sequence), return the corresponding features.
        using KeyMap.ngram_key to get key
        Args:
            ngram: a ngram (Seq())

        Returns:
            a list of feature ids
        """
        ids = []
        for m in self.map_list:
            key = m.ngram_key(ngram)
            if key is not None:
                id = self.trie.find(key)
                if id is not None:
                    ids.append(id)
        return ids

    def level_ngram_find(self, seq, level, pos, order):
        ids = []
        for m in self.map_list:
            key = m.level_ngram_key(seq, level, pos, order)
            if key is not None:
                id = self.trie.find(key)
                if id is not None:
                    ids.append(id)
        return ids

    def write(self, f, values=None):
        f.write('==========================\n')
        write_value(f, self.type, 'feat_type')
        write_value(f, self.cut_off, 'cut_off')
        write_value(f, self.map_tag_to_level, 'map_tag')
        write_value(f, self.num, 'feat_num')

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
        f.readline()
        type = read_value(f)
        cut_off = read_value(f)
        map_tag = read_value(f)
        self.__init__(type, cut_off, map_tag)

        self.num = read_value(f)
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
