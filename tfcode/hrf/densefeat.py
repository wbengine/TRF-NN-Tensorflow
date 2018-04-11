import os
import json
from multiprocessing import Process, Manager, Queue, Value

from base import *
from trf.common import feat2 as feat


class Feats(object):
    def __init__(self, type_dict, map_tag_to_level={'w': 0, 'c': 1},
                 dense_level='c', dense_vocab_size=100):

        self.type_dict = type_dict
        self.map_tag_to_level = map_tag_to_level
        self.dense_level = dense_level
        self.dense_vocab_size = dense_vocab_size

        self.feat_list = []
        for key, v in type_dict.items():
            self.feat_list.append(SingleFeat(key, v, map_tag_to_level,
                                             dense_level=dense_level,
                                             dense_vocab_size=dense_vocab_size))
        self.num = 0
        self.values = []
        self.create_values_buf(0)

    def get_order(self):
        return max([f.get_order() for f in self.feat_list])

    def create_values_buf(self, feat_num):
        self.num = feat_num
        self.values = np.zeros(feat_num)

    def load_from_seqs(self, seqs):
        self.num = 0
        for ftype in self.feat_list:
            self.num = ftype.create_from_seqs(seqs, self.num)

        self.create_values_buf(self.num)
        # print('[{}.{}] Load features {:,}'.format(__name__, self.__class__.__name__, self.num))

    def save(self, f, value_buf=None):
        if value_buf is None:
            value_buf = self.values

        feat.write_value(f, self.type_dict, 'feature_type')
        feat.write_value(f, self.map_tag_to_level, 'map_tag_to_level')
        feat.write_value(f, self.dense_level, 'dense_level')
        feat.write_value(f, self.dense_vocab_size, 'dense_vocab_size')
        feat.write_value(f, self.num, 'feature_total_num')
        feat.write_value(f, self.get_order(), 'feature_max_order')
        for ftype in self.feat_list:
            ftype.write(f, value_buf)

    def restore(self, f):
        type_dict = feat.read_value(f)
        map_tag = feat.read_value(f)
        dense_level = feat.read_value(f)
        dense_vocab = feat.read_value(f)
        self.__init__(type_dict, map_tag, dense_level, dense_vocab)

        self.num = feat.read_value(f)
        feat.read_value(f)
        self.create_values_buf(self.num)
        for ftype in self.feat_list:
            ftype.read(f, self.values)

    @staticmethod
    def load(fp):
        type_dict = feat.read_value(fp)
        map_tag = feat.read_value(fp)
        dense_level = feat.read_value(fp)
        dense_vocab = feat.read_value(fp)
        f = Feats(type_dict, map_tag, dense_level, dense_vocab)

        f.num = feat.read_value(fp)
        f.max_order = feat.read_value(fp)
        f.create_values_buf(f.num)
        for ftype in f.feat_list:
            ftype.read(fp, f.values)
        return f

    def seq_find(self, seq, depend_on=None):
        return sum([ftype.seq_find(seq, depend_on) for ftype in self.feat_list], [])

    def seq_weight(self, seq, depend_on=None):
        return np.sum(self.values[self.seq_find(seq, depend_on)])

    def ngram_find(self, seq, pos, order):
        return sum([ftype.ngram_find(seq, pos, order) for ftype in self.feat_list], [])

    def ngram_weight(self, seq, pos, order):
        return np.sum(self.values[self.ngram_find(seq, pos, order)])

    def ngram_enumerate(self, seq, pos, order):
        vec = np.zeros(self.dense_vocab_size ** order)
        for f in self.feat_list:
            vec += f.ngram_enumerate(seq, pos, order, self.values)
        return vec

    def ngram_enumerate_ids(self, seq, pos, order):
        """
        Returns:
            np.array of shape [self.dense_vocab_size ** order, features_num]
        """
        ids = np.zeros([self.dense_vocab_size ** order, 0], dtype='int32')
        for f in self.feat_list:
            a = f.ngram_enumerate(seq, pos, order, None)
            ids = np.concatenate([ids, a], axis=1)
        return ids

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
    def __init__(self, type_dict, map_tag_to_level={'w': 0, 'c': 1},
                 dense_level='c', dense_vocab_size=100, sub_process_num=4):

        super().__init__(type_dict, map_tag_to_level, dense_level, dense_vocab_size)

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
    def __init__(self, type, map_tag_to_level, dense_level, dense_vocab_size):
        """
        Args:
            type: Single Type, such as w[1], w[2], w[1]c[1] as so on
        """
        self.dense_level = dense_level
        self.dense_vocab_size = dense_vocab_size
        self.map_tag_to_level = map_tag_to_level
        self.key_map, self.idx_map = self.parse(type)

    def get_order(self):
        """return the order for each level"""
        pos = self.idx_map[1]
        dense_order = pos[-1] - pos[0] + 1
        return dense_order

    def get_idx_size(self):
        return self.dense_vocab_size ** self.get_order()

    def parse(self, ftype):
        """
        Get the detail of the KeyMap, such as:
            if w->0, c->1, dense_level=c=1
            w[1]c[1]w[1] -> key_map = ( level=[0, 0], pos=[-1, 1] ), index_map= ( level=[1], pos=[0] )
        Args:
            ftype: str, type str

        Returns:
            (key_map_level, key_map_pos), (index_map_level, index_map_pos)
        """
        if ftype.find(':') != -1:
            raise TypeError('[%s.%s] only support signle type!' % (__name__, self.__class__.__name__))

        a = filter(None, ftype.split(']'))
        n = 0
        key_map_level = []
        key_map_pos = []
        idx_map_level = []
        idx_map_pos = []
        for s in a:
            tags, nums = s.split('[')
            i = int(nums)
            pos = list(range(n, n+i))

            for tag in tags:
                if tag in self.map_tag_to_level:
                    cur_level = self.map_tag_to_level[tag]
                    cur_map_level = [cur_level] * len(pos)
                    if cur_level == self.dense_level:
                        idx_map_level += cur_map_level
                        idx_map_pos += pos
                    else:
                        key_map_level += [self.map_tag_to_level[tag]] * len(pos)
                        key_map_pos += pos

            n += i

        if len(idx_map_pos) == 0:
            raise TypeError('[%s.%s] cannot find the dense level in feat %s' %
                            (__name__, self.__class__.__name__, type))

        center_pos = idx_map_pos[0]

        key_map_level = np.array(key_map_level)
        key_map_pos = np.array(key_map_pos) - center_pos
        idx_map_level = np.array(idx_map_level)
        idx_map_pos = np.array(idx_map_pos) - center_pos

        return (key_map_level.tolist(), key_map_pos.tolist()), (idx_map_level.tolist(), idx_map_pos.tolist())

    def ngram_key(self, seq, pos, order):
        """
        input a Seq() and find the key based the dense-level
        """
        if self.idx_map[1][-1] + 1 > order or \
                self.key_map[1][0] + pos < 0 or self.key_map[1][-1] + pos >= len(seq):
            return None, None

        key = [seq.x[i, j + pos] for i, j in zip(*self.key_map)]
        idx = [seq.x[i, j + pos] for i, j in zip(*self.idx_map)]
        idx = sp.map_list(idx, self.dense_vocab_size)
        return key, idx

        # key_pos = self.key_map[1] + pos
        # idx_pos = self.idx_map[1] + pos
        #
        # if self.idx_map[1][-1] + 1 > order:
        #     return None, None
        #
        # if key_pos[0] < 0 or key_pos[-1] >= len(seq):
        #     return None, None
        #
        # key = seq.x[self.key_map[0], key_pos]
        # idx = seq.x[self.idx_map[0], idx_pos]
        # idx = sp.map_list(idx, self.dense_vocab_size)
        # return key, idx

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

        key_idx_list = []
        order = self.idx_map[1][-1] + 1

        if depend_on is None:
            for i in range(0, len(seq)-order+1):
                key, idx = self.ngram_key(seq, i, order)
                if key is not None and idx is not None:
                    key_idx_list.append((key, idx))

        elif isinstance(depend_on, tuple) and \
            depend_on[0] == self.dense_level and \
            depend_on[1] is not None:

            pos = depend_on[1]
            for i in range(max(0, pos-order+1), min(pos+1, len(seq)-order+1)):
                key, idx = self.ngram_key(seq, i, order)
                if key is not None and idx is not None:
                    key_idx_list.append((key, idx))

        else:
            raise TypeError('[{}.{}] get_keys() cannot support the input depend_on={}'.format(
                __name__, self.__class__.__name__, depend_on))

        return key_idx_list


class SingleFeat(object):
    def __init__(self, type='', cut_off=None, map_tag_to_level={'w': 0, 'c': 1},
                 dense_level='c', dense_vocab_size=100):
        """
        save the dense_level into a arrray, not in the tree.
        The cases sharing one trie structure: w[1:3]c[1]
        """

        self.type = type
        self.cut_off = cut_off
        self.map_tag_to_level = map_tag_to_level
        self.dense_level = dense_level if isinstance(dense_level, int) else map_tag_to_level[dense_level]
        self.dense_vocab_size = dense_vocab_size
        self.dense_idx_size = 0
        if type != '':
            self.map_list, self.cut_list = self.analyze_type(type)
            # self.max_order = max([m.get_order() for m in self.map_list])
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
        cut_list = []
        for i, t in enumerate(type_list):
            map_list.append(KeyMap(t, self.map_tag_to_level, self.dense_level, self.dense_vocab_size))
            if self.cut_off is None:
                cut_list.append(0)
            elif isinstance(self.cut_off, list):
                cut_list.append(self.cut_off[i])
            else:
                cut_list.append(self.cut_off)

        a = [m.get_idx_size() for m in map_list]
        assert np.all(np.array(a) == a[0])
        self.dense_idx_size = a[0]

        return map_list, cut_list

    def get_order(self):
        return max([m.get_order() for m in self.map_list])

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
                keys_idxs_tuple = key_map.get_keys(seq)
                for k, idx in keys_idxs_tuple:
                    count = count_trie.setdefault(k, 0)
                    count.data += 1
                    if count.data >= cut_num:
                        # add the feature
                        id = self.trie.setdefault(k, beg_id)
                        if id.data == beg_id:
                            beg_id += self.dense_idx_size
                            self.num += self.dense_idx_size
        return beg_id

    def seq_find(self, seq, depend_on=None):
        ids = []
        for m in self.map_list:
            for key, idx in m.get_keys(seq, depend_on):
                id = self.trie.find(key)
                if id is not None:
                    ids.append(id + idx)
        return ids

    def ngram_find(self, seq, pos, order):
        ids = []
        for m in self.map_list:
            key, idx = m.ngram_key(seq, pos, order)
            if key is not None:
                id = self.trie.find(key)
                if id is not None:
                    ids.append(id + idx)
        return ids

    def ngram_enumerate(self, seq, pos, order, values=None):
        # used to compute the emission matrix in forword-backword algorithm
        # if values is None, then return the feature ids
        vec = np.zeros(self.dense_vocab_size ** order)
        ids = np.zeros([self.dense_vocab_size ** order, 0], dtype='int32')
        for m in self.map_list:
            key, _ = m.ngram_key(seq, pos, order)
            if key is None:
                continue
            buf_beg = self.trie.find(key)
            if buf_beg is None:
                continue

            if values is not None:
                v = values[buf_beg: buf_beg + self.dense_idx_size]
                vec += np.repeat(v, self.dense_vocab_size ** (order - self.get_order()))
            else:
                v = np.arange(buf_beg, buf_beg + self.dense_idx_size)
                v = np.repeat(v, self.dense_vocab_size ** (order - self.get_order()))
                ids = np.concatenate([ids, np.reshape(v, [-1, 1])], axis=1)

        if values is not None:
            return vec
        else:
            return ids

    def write(self, f, values=None):
        f.write('==========================\n')
        feat.write_value(f, self.type, 'feat_type')
        feat.write_value(f, self.cut_off, 'cut_off')
        feat.write_value(f, self.map_tag_to_level, 'map_tag')
        feat.write_value(f, self.dense_level, 'dense_level')
        feat.write_value(f, self.dense_vocab_size, 'dense_vocab_size')
        feat.write_value(f, self.num, 'feat_num')

        write_num = 0
        for key, id in trie.TrieIter(self.trie):
            if id is None:
                continue

            if values is not None:
                v = values[id: id + self.dense_idx_size].tolist()
            else:
                v = None
            f.write('key={} id={} value={}\n'.format(json.dumps(key), id, json.dumps(v)))
            write_num += self.dense_idx_size

        assert write_num == self.num

    def read(self, f, values=None):
        f.readline()
        type = feat.read_value(f)
        cut_off = feat.read_value(f)
        map_tag = feat.read_value(f)
        dense_level = feat.read_value(f)
        dense_vocab_size = feat.read_value(f)
        self.__init__(type, cut_off, map_tag, dense_level=dense_level, dense_vocab_size=dense_vocab_size)

        self.num = feat.read_value(f)
        for i in range(self.num // self.dense_idx_size):
            s = f.__next__()
            s = s.replace('key=', '|')
            s = s.replace(' id=', '|')
            s = s.replace(' value=', '|')
            a = list(filter(None, s.split('|')))
            key = json.loads(a[0])
            id = int(a[1])
            v = json.loads(a[2])  # a list
            self.trie.insert(key, id)
            if values is not None:
                values[id: id+self.dense_idx_size] = np.array(v) if isinstance(v, list) else v  # to array
