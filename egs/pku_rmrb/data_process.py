import os
import json
import codecs

from base import *


pku_rmrb_path = '//166.111.64.114/workspace2/spmiData/pku_rmrb'
output_path = 'data'

sentence_max_len = 100
sentence_min_len = 1
sentence_end_tokens = ['，', '。', '：', '；', '？', '！',
                       ',', '.', ':', ';', '?', '!']
sentence_following_tokens = ['”', '’']


def process_entity(token_list):
    stack = []
    res_token_list = []
    for i, token in enumerate(token_list):
        a = token.split('/')
        if len(a) != 2:
            raise TypeError('error token = {}'.format(token))
        w = a[0]
        t = a[1]

        if w.find('[') == 0:
            stack.append(i)
            res_token_list.append(token[1:])  # remove '['
        elif t.find(']') != -1:
            t, ne_tag = t.split(']')
            res_token_list.append(w + '/' + t)

            if stack:
                beg = stack.pop()
        else:
            res_token_list.append(token)

    return res_token_list


def process_line(line):
    # input a line of the source files
    # output separated char sequences and tag sequences
    token_list = line.split()

    # process the name_entity
    token_list = process_entity(token_list)

    sent_list = []  # the separated sentences

    word_seq = []
    tag_seq = []
    sent_end_flag = False
    for token in token_list[1:]:
        a = token.split('/')
        if len(a) != 2:
            raise TypeError('error token = {}'.format(token))
        w = a[0]
        t = a[1].lower()

        # process the sentence-end tokens
        if sent_end_flag and w not in sentence_following_tokens + sentence_end_tokens:
            sent_list.append((list(word_seq), list(tag_seq)))
            word_seq = []
            tag_seq = []
            sent_end_flag = False

        word_seq.append(w)
        tag_seq.append(t)

        if w in sentence_end_tokens:
            sent_end_flag = True

    sent_list.append((list(word_seq), list(tag_seq)))

    # word to chars
    char_sent_list = []
    for word_seq, tag_seq in sent_list:
        char_seq = []
        char_tag_seq = []
        for w, t in zip(word_seq, tag_seq):

            w = w.split('{')[0]  # remove the pronounce, such as '这{zhe4}'

            for i, c in enumerate(list(w)):
                char_seq.append(c)

                if len(w) == 1:
                    poc = 's'
                elif i == 0:
                    poc = 'b'
                elif i == len(w) -1:
                    poc = 'e'
                else:
                    poc = 'm'
                char_tag_seq.append(t + '_' + poc)
        char_sent_list.append((char_seq, char_tag_seq))

    return char_sent_list


def process_data(input_file, output_file):
    # process the source files
    # separate the word and the tags
    # split the word to characters
    truncated_sent_num = 0
    total_sent_num = 0
    with open(input_file, 'rt', errors='ignore') as fin, \
         open(output_file + '.chr', 'wt') as fchar, \
         open(output_file + '.tag', 'wt') as ftag:
        for line in fin:
            if not line.split():
                continue

            try:
                char_sent_list = process_line(line)
                for char_seq, tag_seq in char_sent_list:
                    if len(char_seq) < sentence_min_len:
                        continue
                    if len(char_seq) > sentence_max_len:
                        truncated_sent_num += 1
                        char_seq = char_seq[0: sentence_max_len]
                        tag_seq = tag_seq[0: sentence_max_len]
                    total_sent_num += 1
                    fchar.write(' '.join(char_seq) + '\n')
                    ftag.write(' '.join(tag_seq) + '\n')

            except Exception as e:
                print('A Error occurs in\n  file: {}\n  line: {}\n'.format(input_file, line))
                print('Error: {}'.format(e))

    print('process_data: total_sent_num=', total_sent_num)
    print('process_data: truncated_sent_num=', truncated_sent_num, '({:.2f}%)'.format(100*truncated_sent_num/total_sent_num))


def main_processing():
    folders = ['1998-01-12-105标记-带音-20030226', '2000-01-12-105标记-带音-20130226']

    source_file_list = []
    output_file_list = []
    for folder in folders:
        source_dir = os.path.join(pku_rmrb_path, folder)
        for fname in os.listdir(source_dir):
            if os.path.splitext(fname)[-1] == '.txt':
                source_file_list.append(os.path.join(source_dir, fname))
                output_file_list.append(os.path.join(output_path + '/txt', os.path.splitext(fname)[0]))

    wb.mkdir(output_path)
    wb.mkdir(output_path + '/txt')
    for source_file, output_file in zip(source_file_list, output_file_list):
        print(source_file, '\t', output_file)
        process_data(source_file, output_file)


def generate_file_list():
    output_file_list = []
    for fname in os.listdir(output_path + '/txt'):
        a = os.path.splitext(fname)[0]
        a = os.path.join(output_path, 'txt', a)
        if a not in output_file_list:
            output_file_list.append(a)

    # generate file list for training and developing set
    dev_files = [output_file_list[11], output_file_list[-1]]
    train_files = output_file_list[0: 11] + output_file_list[12:-1]

    return train_files, dev_files


def generate_vocab(file_list, write_vocab, cutoff=0, add_beg_token='<s>', add_end_token='</s>', add_unk_token='<unk>'):
    v = vocab.Vocab().generate_vocab(file_list, cutoff=cutoff,
                                     add_beg_token=add_beg_token,
                                     add_end_token=add_end_token,
                                     add_unk_token=add_unk_token)

    v.write(write_vocab)


def count_length(file_list):

    count_dict = dict()

    for file in file_list:
        print('length_count:', file)
        with open(file + '.chr', 'rt') as f:
            for line in f:
                n = len(line.split())
                if n > 0:
                    count_dict.setdefault(n, 0)
                    count_dict[n] += 1

    min_len = min(count_dict.keys())
    max_len = max(count_dict.keys())

    print('min_len=', min_len)
    print('max_len=', max_len)

    count_list = [0] * (max_len+1)
    for n, count in count_dict.items():
        count_list[n] = count

    return count_list


def generate_envir(train_list, dev_list, output_dir):
    # generate vocab
    generate_vocab([s + '.chr' for s in train_list], os.path.join(output_dir, 'vocab.chr.txt'))
    generate_vocab([s + '.tag' for s in train_list], os.path.join(output_dir, 'vocab.tag.txt'))

    # count length distributions
    len_count = count_length(train_list)

    # write infos
    infos = {'train': [(x + '.chr', x + '.tag') for x in train_list],
             'dev': [(x + '.chr', x + '.tag') for x in dev_list],
             'len_count': len_count,
             'vocab': (os.path.join(output_dir, 'vocab.chr.txt'),
                       os.path.join(output_dir, 'vocab.tag.txt'))}
    with open(os.path.join(output_dir, 'data_info.txt'), 'w') as f:
        f.write(json.dumps(infos, ensure_ascii=False, indent=4))


def generate_demo(train_list, dev_list):
    demo_dir = 'demo'
    wb.mkdir(demo_dir)

    demo_train = demo_dir + '/train'
    demo_valid = demo_dir + '/dev'

    demo_train_line = 2
    demo_valid_line = 1

    def copy_from_beg(src_file, tar_file, line_num):
        with open(src_file, 'rt') as fsrc, open(tar_file, 'wt') as ftar:
            for line, _ in zip(fsrc, range(line_num)):
                ftar.write(line)

    copy_from_beg(train_list[0] + '.chr', demo_train + '.chr', demo_train_line)
    copy_from_beg(train_list[0] + '.tag', demo_train + '.tag', demo_train_line)
    copy_from_beg(dev_list[0] + '.chr', demo_valid + '.chr', demo_valid_line)
    copy_from_beg(dev_list[0] + '.tag', demo_valid + '.tag', demo_valid_line)

    generate_envir([demo_train], [demo_valid], demo_dir)


if __name__ == '__main__':
    # main_processing()

    # separate train/dev
    train_list, dev_list = generate_file_list()
    print('train_list=')
    print(json.dumps(train_list, indent=2))
    print('dev_list=')
    print(json.dumps(dev_list, indent=2))

    # generate the config
    # generate_envir(train_list, dev_list, output_path)

    # generate a demo set
    generate_demo(train_list, dev_list)



