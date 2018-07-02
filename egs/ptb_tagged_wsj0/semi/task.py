import json
import os
from base import *
# please run ./full/data_process.py first


def get_first_lines(input_file, output_file, num):
    with open(input_file, 'rt') as fin, open(output_file, 'wt') as fout:
        n = 0
        for line in fin:
            fout.write(line)
            n += 1
            if n >= num:
                break


def read_txt(file):
    with open(file, 'rt') as f:
        return [line for line in f]


def write_txt(file, str_list):
    with open(file, 'wt') as f:
        for s in str_list:
            f.write(s)


def add_path(old_paths):
    return [os.path.join('../full', s) for s in old_paths]

if __name__ == '__main__':
    with open('../full/data.info') as f:
        info = json.load(f)

    with open('../../google1B/data.info') as f:
        info_ext = json.load(f)

    info['train'] = [add_path(info['train'][0])]
    info['valid'] = [add_path(info['valid'][0])]
    info['test'] = [add_path(info['test'][0])]
    info['vocab'] = add_path(info['vocab'])
    info['google_1b'] = [(s, None) for s in info_ext['train_all']]

    rate = 0.02
    train_wod_list = read_txt(info['train'][0][0])
    train_tag_list = read_txt(info['train'][0][1])

    for n in [100, 500, 1000, 5000, len(train_wod_list)]:
        file_name = wb.mkdir('data/') + 'train.%d' % n
        write_txt(file_name + '.wod', train_wod_list[0: n])
        write_txt(file_name + '.tag', train_tag_list[0: n])

        part_n = int(n / rate)

        print('n=%d, part_n=%d Load 1billion data!' % (n, part_n))
        train_wod_list_ex = read_txt(info_ext['train_all'][0])
        fi = 1
        while len(train_wod_list_ex) < part_n:
            train_wod_list_ex += read_txt(info_ext['train_all'][fi])
            fi += 1
        assert len(train_wod_list_ex) >= part_n

        write_txt(file_name + '.part.wod', train_wod_list_ex[0: part_n])

        info['train%d' % n] = [(file_name + '.wod', file_name + '.tag')]
        info['train%d.part' % n] = [(file_name + '.part.wod', None)]

    with open('data.info', 'wt') as f:
        json.dump(info, f, indent=4)