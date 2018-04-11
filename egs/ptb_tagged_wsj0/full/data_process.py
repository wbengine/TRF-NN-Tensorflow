import os
import json
import re
import zipfile

from base import *

if wb.is_window():
    ptb_zip_file = '//166.111.64.114/workspace2/spmiData/LDC_tgz/treebank_3_LDC99T42_20161007.zip'
else:
    ptb_zip_file = '/home/wangbin/NAS_workspace2/spmiData/LDC_tgz/treebank_3_LDC99T42_20161007.zip'
output_path = '../data'

wb.mkdir(output_path)


def get_src_files():
    tag_path = os.path.join(output_path,
                            'treebank_3_LDC99T42_20161007/treebank_3_LDC99T42_20161007/treebank_3/tagged/pos/wsj/')

    if not wb.exists(tag_path):
        with wb.processing('unpacking the zip file'):
            zfile = zipfile.ZipFile(ptb_zip_file)
            for names in zfile.namelist():
                zfile.extract(names, output_path)

    train_list = []
    valid_list = []
    test_list = []
    for session in ['{:0>2}'.format(i) for i in range(25)]:
        for file in sorted(os.listdir(os.path.join(tag_path, session))):
            if int(session) <= 20:
                train_list.append(os.path.join(tag_path, session, file))
            elif int(session) <= 22:
                valid_list.append(os.path.join(tag_path, session, file))
            else:
                test_list.append(os.path.join(tag_path, session, file))

    return train_list, valid_list, test_list


def process_file(src_file, fword, ftag, word_vocab=None, tag_vocab=None):
    """
    read from the source file and write to fword and ftag
    Args:
        src_file: str, file name
        fword: fp, word file
        ftag:  fp, tag file
        word_vocab: word Vocab
        tag_vocab:  tag Vocab
    """
    with open(src_file, 'rt') as f:
        word_list = []
        tag_list = []
        for line in f:
            a = line.split()
            if line[0:5] == '=====' or len(a) == 0:
                if word_list:
                    if word_list != ['\'\'']:
                        fword.write(' '.join(word_list) + '\n')
                        ftag.write(' '.join(tag_list) + '\n')
                    word_list = []
                    tag_list = []
            else:
                for token in a:
                    res = process_token(token, word_vocab, tag_vocab)
                    if res is not None:
                        word_list.append(res[0])
                        tag_list.append(res[1])

        if word_list:
            if word_list != ['\'\'']:
                fword.write(' '.join(word_list) + '\n')
                ftag.write(' '.join(tag_list) + '\n')


def process_token(token, word_vocab=None, tag_vocab=None):
    """input token and return the word and tag, or None"""
    if token.find('/') == -1:
        return None

    try:
        i = token.rfind('/')
        w = token[0: i]
        t = token[i+1:].split('|')[0]

        w = w.lower()
        t = t.upper()
    except ValueError:
        print('token=', token)
        raise ValueError

    if t == 'CD' and len(re.findall('[a-z]', w)) == 0:  # number
        w = 'n'
    # if t in [',', '.', ':', '?', '$', '#', '\'\'', '``', '(', ')']:
    #     return None

    if word_vocab is not None:
        if w not in word_vocab:
            w = '<unk>'

    if tag_vocab is not None:
        if t not in tag_vocab:
            t = '<unk>'

    return w, t


def process_data(file_list, output_name, word_vocab=None, tag_vocab=None):
    with open(output_name + '.wod', 'wt') as fwod, open(output_name + '.tag', 'wt') as ftag:
        for src_file in file_list:
            process_file(src_file, fwod, ftag, word_vocab, tag_vocab)


if __name__ == '__main__':

    train_files, valid_files, test_files = get_src_files()

    # summary all the files
    whole_data = os.path.join(output_path, 'src_data')
    process_data(train_files + valid_files + test_files, whole_data)

    # generate vocabularys
    v_wod = vocab.Vocab().generate_vocab([whole_data + '.wod'], max_size=10000+1,
                                         add_beg_token='<s>',
                                         add_end_token='</s>',
                                         add_unk_token='<unk>')
    v_wod.write(os.path.join(output_path, 'vocab.wod'))

    v_tag = vocab.Vocab().generate_vocab([whole_data + '.tag'],
                                         add_beg_token='<s>',
                                         add_end_token='</s>',
                                         add_unk_token=None)
    v_tag.write(os.path.join(output_path, 'vocab.tag'))

    # output the final files
    output_train = os.path.join(output_path, 'train')
    output_valid = os.path.join(output_path, 'valid')
    output_test = os.path.join(output_path, 'test')
    process_data(train_files, output_train, v_wod, v_tag)
    process_data(valid_files, output_valid, v_wod, v_tag)
    process_data(test_files, output_test, v_wod, v_tag)

    # write information
    info = dict()
    info['train'] = [(output_train + '.wod', output_train + '.tag')]
    info['valid'] = [(output_valid + '.wod', output_valid + '.tag')]
    info['test'] = [(output_test + '.wod', output_test + '.tag')]
    info['vocab'] = (os.path.join(output_path, 'vocab.wod'), os.path.join(output_path, 'vocab.tag'))
    info['nbest'] = reader.wsj0_nbest()
    with open('data.info', 'wt') as f:
        json.dump(info, f, indent=4)

    # data = reader.Data().load_raw_data([info['train'][0][0], info['valid'][0][0], info['test'][0][0]])
    # print(data.get_vocab_size())
    #
    # txtinfo = wb.TxtInfo(info['train'][0][0])
    # print(txtinfo)

