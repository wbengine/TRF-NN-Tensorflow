import os
import numpy as np
from collections import OrderedDict

from base import wblib as wb


class TxtscanData(object):
    def __init__(self):
        self.legend_list = []
        self.dict_data_list = []

    def add(self, legend, dict_data_or_names, datas=None):
        # add(legend, {'x': [1,2,3], 'y': [4,5,6]})
        # add(legend, ['x', 'y'], [[1,2,3], [4,5,6]])
        if datas is None:
            dict_data = dict_data_or_names
        else:
            dict_data = OrderedDict([(s, d) for s, d in zip(dict_data_or_names, datas)])

        self.legend_list.append(legend)
        self.dict_data_list.append(dict_data)

    def write(self, fname):
        with open(fname, 'wt') as f:
            f.write('[head]\n')
            i = 1
            for legend, dict_data in zip(self.legend_list, self.dict_data_list):
                f.write('{} = {}:{}\n'.format(legend, i, i+len(dict_data)-1))
                i += len(dict_data)

            f.write('\n[data]\n')
            key_all = []
            data_all = []
            for legend, dict_data in zip(self.legend_list, self.dict_data_list):
                keys = dict_data.keys()
                datas = [dict_data[k] for k in keys]
                key_all += keys
                data_all += datas

            # align data
            max_n = 0
            for d in data_all:
                max_n = max(max_n, len(d))

            write_data = []
            for d in data_all:
                if len(d) < max_n:
                    write_data.append(list(d) + [-1]*(max_n - len(d)))
                else:
                    write_data.append(d)

            f.write(' '.join(['{:<10}'.format(k) for k in key_all]) + '\n')
            for v_tuple in zip(*write_data):
                f.write(' '.join(['{:<10.4f}'.format(v) for v in v_tuple]) + '\n')



