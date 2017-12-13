import os
import subprocess
import re
import time

from . import wblib as wb

err_str = 'cannot find the path of stanford parser! \n' \
          'Please download the stanford parser from ' \
          '\"https://nlp.stanford.edu/software/lex-parser.shtml\" \n' \
          'Then set environment variable STANFORD_PARSER_HOME to the parser path. '


class Info(object):
    def __init__(self, out_line=None, label_vocab=None, word_vocab=None):
        self.label = 'compound'
        self.label_id = -1
        self.parent = '<unk>'
        # self.parent_pos = 0
        self.parent_id = -1
        self.child = '<unk>'
        # self.child_pos = 0
        self.child_id = -1

        if out_line is not None:
            try:
                a = list(filter(None, re.split('[(, )]', out_line)))
                self.label = a[0]
                sep = a[1].rfind('-')
                self.parent = self.word_precess(a[1][0:sep])
                # self.parent_pos = int(a[1][sep+1:])
                sep = a[2].rfind('-')
                self.child = self.word_precess(a[2][0:sep])
                # self.child_pos = int(a[2][sep+1:])
            except:
                print(out_line)
                print(a)

        try:
            if label_vocab is not None:
                self.set_label_vocab(label_vocab)

            if word_vocab is not None:
                self.set_word_vocab(word_vocab)

        except KeyError:
            print(out_line)
            print(a)
            print(str(self))
            raise KeyError

    def word_precess(self, w):
        if w == 'ROOT':
            return '<s>'
        return w

    def set_word_vocab(self, word_vocab):
        self.parent_id = word_vocab[self.parent]
        self.child_id = word_vocab[self.child]

    def set_label_vocab(self, label_vocab):
        self.label_id = label_vocab[self.label]

    def __str__(self):
        return '{}[id={}]  {}[id={}] -> {}[id={}]'.format(
            self.label, self.label_id, self.parent, self.parent_id,
            self.child, self.child_id
        )


def dependency_parser(text_list, label_to_id=None, word_to_id=None, logdir=None):
    tbeg = time.time()

    if 'STANFORD_PARSER_HOME' not in os.environ:
        raise EnvironmentError(err_str)

    parser_dir = os.environ['STANFORD_PARSER_HOME']
    parser_class = 'edu.stanford.nlp.parser.lexparser.LexicalizedParser'
    parser_model = 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
    parser_options = '-sentences newline -tokenized -outputFormat \"typedDependencies\"'

    cmd_list = ['java -mx150m']
    cmd_list += ['-cp', '\"%s\"' % os.path.join(parser_dir, '*')]
    cmd_list.append(parser_class)
    cmd_list.append(parser_options)
    cmd_list.append(parser_model)
    cmd_list.append('-')

    p = subprocess.Popen(' '.join(cmd_list),
                         shell=True,
                         universal_newlines=True,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    out, err = p.communicate('\n'.join([' '.join(s) for s in text_list]))

    out_list = filter(None, out.split('\n\n'))
    err_list = []
    for line in err.splitlines():
        if line.find('Parsing [sent.') == 0:
            err_list.append(line)

    out_infos = [[Info(s, label_vocab=label_to_id, word_vocab=word_to_id) for s in out.split('\n')] for out in out_list]

    if logdir is not None:
        with open(os.path.join(logdir, 'parser.log'), 'at') as f:
            f.write('dependency_parser: seq_num={} time={:.2f} s\n'.format(
                len(text_list), time.time()-tbeg))

    return out_infos, err_list

if __name__ == '__main__':

    import reader
    import time

    # data = reader.Data().load_raw_data(reader.ptb_raw_dir(), add_beg_token='<s>',
    #                                    add_end_token='</s>')
    # text_list = data.seqs_to_text(data.datas[0], skip_beg_token=True, skip_end_token=True)

    # out_list, err_list = dependency_parser(text_list, word_to_id=data.word_to_id)
    #
    # outdir = 'debug'
    # wb.mkdir(outdir)
    # with open(os.path.join(outdir, 'ptb.parser.res'), 'wt') as f:
    #     for out, err in zip(out_list, err_list):
    #         f.write(err + '\n')
    #         for info in out:
    #             f.write(str(info) + '\n')

    data = reader.Data().load_raw_data(['ptb_demo/ptb.train.txt'], add_beg_token='<s>',
                                       add_end_token='</s>')
    text_list = data.seqs_to_text(data.datas[0], skip_beg_token=True, skip_end_token=True)[0:100]

    print('begin parser....')
    tbeg = time.time()
    out_list, err_list = dependency_parser(text_list, word_to_id=data.word_to_id)
    print('end, time={}s'.format(time.time() - tbeg))

    outdir = 'debug'
    wb.mkdir(outdir)
    with open(os.path.join(outdir, 'ptb.parser.res'), 'wt') as f:
        for out, err in zip(out_list, err_list):
            print(err)
            f.write(err + '\n')
            for info in out:
                print(str(info))
                f.write(str(info) + '\n')
