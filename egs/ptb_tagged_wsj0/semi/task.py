import json
from base import *
# please run ../full/data_process.py first


def get_first_lines(input_file, output_file, num):
    with open(input_file, 'rt') as fin, open(output_file, 'wt') as fout:
        n = 0
        for line in fin:
            fout.write(line)
            n += 1
            if n >= num:
                break

if __name__ == '__main__':
    with open('../full/data.info') as f:
        info = json.load(f)

    for n in [100, 1000, 5000]:
        file_name = wb.mkdir('data/') + 'train.%d' % n
        get_first_lines(info['train'][0][0], file_name + '.wod', n)
        get_first_lines(info['train'][0][1], file_name + '.tag', n)

        info['train%d' % n] = [(file_name + '.wod', file_name + '.tag')]

    with open('data.info', 'wt') as f:
        json.dump(info, f, indent=4)