import numpy as np
import time
import threading

class node:
    def __init__(self):
        self.sub = dict()
        self.data = None

    def insert(self, key_list, data=None):
        """
        Insert a value. If the corresponding node exits, this method will revise the self.data to the input data.
        If you donot want to revise the data, use method self.setdefault()
        :param key_list: a list of key
        :param data: the value
        :return: return the node
        """
        if len(key_list) == 0:
            self.data = data
            return self

        key = key_list[0]
        if key not in self.sub:
            sub_node = node()
            self.sub[key] = sub_node
        else:
            sub_node = self.sub[key]

        return sub_node.insert(key_list[1:], data)

    def setdefault(self, key_list, data=None):
        """
        Set the default value.
        If the node exits, this method return the data.
        If not, this method will set the self.data to the input data and return the input data
        :param key_list:  key list
        :param data: input data
        :return: return the found node
        """
        if len(key_list) == 0:
            if self.data is None:
                self.data = data
            return self

        key = key_list[0]
        if key not in self.sub:
            sub_node = node()
            self.sub[key] = sub_node
        else:
            sub_node = self.sub[key]

        return sub_node.setdefault(key_list[1:], data)

    def find_node(self, key_list):
        """find a node and return the node"""
        if len(key_list) == 0:
            return self

        if key_list[0] in self.sub:
            return self.sub[key_list[0]].find_node(key_list[1:])

        return None

    def find(self, key_list):
        """find a node the return the data"""
        fnode = self.find_node(key_list)
        if fnode is not None:
            return fnode.data
        return None


class trie_iter:
    def __init__(self, head, key_list=list(), is_sorted=False):
        if is_sorted:
            self.iter = iter(sorted(head.sub.items(), key=lambda x: x[0]))
        else:
            self.iter = iter(head.sub.items())
        self.sub_trie_iter = None
        self.key_list = key_list
        self.is_sorted = is_sorted

    def __next__(self):
        if self.sub_trie_iter is not None:
            try:
                return self.sub_trie_iter.__next__()
            except StopIteration:
                self.sub_trie_iter = None

        if self.sub_trie_iter is None:
            try:
                sub_key, sub_trie = self.iter.__next__()
                self.sub_trie_iter = trie_iter(sub_trie, self.key_list + [sub_key], self.is_sorted)
                return self.key_list + [sub_key], sub_trie
            except StopIteration:
                raise StopIteration

    def __iter__(self):
        return self


class level_iter:
    def __init__(self, trie_root, level, key_list=list(), is_sorted=False):
        self.trie_root = trie_root
        self.level = level                       # the level need to iter
        if is_sorted:
            self.iter = iter(sorted(iter(trie_root.sub.items()), key=lambda x:x[0]))  # iter the sub trie
        else:
            self.iter = iter(trie_root.sub.items())  # iter the sub trie
        self.sub_level_iter = None               # the sub level iter
        self.key_list = key_list
        self.is_sorted = is_sorted

    def __next__(self):
        if self.level == 0:
            return self.key_list, self.trie_root
        elif self.level == 1:
            sub_key, sub_trie = self.iter.__next__()
            return self.key_list + [sub_key], sub_trie

        if self.sub_level_iter is not None:
            try:
                return self.sub_level_iter.__next__()
            except StopIteration:
                self.sub_level_iter = None

        while self.sub_level_iter is None:
            try:
                sub_key, sub_trie = self.iter.__next__()
            except StopIteration:
                raise StopIteration

            self.sub_level_iter = level_iter(sub_trie, self.level-1, self.key_list + [sub_key], self.is_sorted)

            try:
                return self.sub_level_iter.__next__()
            except StopIteration:
                self.sub_level_iter = None


    def __iter__(self):
        return self


def search_epoch(trie, file, max_order):
    total_count = 0
    with open(file, 'rt') as f:
        for line in f:
            a = [-1] + [int(i) for i in line.split()] + [-2]
            n = len(a)
            for order in range(1, max_order+1):
                for pos in range(0, n-order+1):
                    count = trie.find(a[pos: pos+order])
                    total_count += count
    print('fun_res=', total_count)


def test():
    file = 'ptb.train.id'
    write = 'ptb.train.py.count'
    max_order = 3

    print('load trie...')
    beg = time.time()
    trie = node()
    with open(file, 'rt') as f:
        for line in f:
            a = [-1] + [int(i) for i in line.split()] + [-2]
            n = len(a)
            for order in range(1, max_order+1):
                for pos in range(0, n-order+1):
                    sub = trie.setdefault(a[pos: pos+order], 0)
                    sub.data += 1
    print('time=', time.time() - beg, 's')

    print('search test...')
    beg = time.time()
    for i in range(2):
        search_epoch(trie, file, max_order)
    print('time=', time.time() - beg, 's')

    print('search test (threading)...')
    beg = time.time()
    thr = []
    for i in range(2):
        thr.append(threading.Thread(target=search_epoch, name=str(i), args=(trie, file, max_order)))
    for t in thr:
        t.start()
    for t in thr:
        t.join()
    print('time=', time.time() - beg, 's')


    # beg = time.time()
    # print 'write...'
    # with open(write, 'wt') as f:
    #     for keys, sub in trie_iter(trie):
    #         f.write(' '.join(str(i) for i in keys) + '\t{}\n'.format(sub.data))
    # print 'time=', time.time() - beg, 's'




if __name__ == '__main__':
    test()
