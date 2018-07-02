

# Note:
# this scripts copy the files generated from kaldi to local folder.
# * Install Kaldi from svn: //166.111.64.101/svn/Gsp/user/wangbin/work/kaldi,
#   which includes scripts developed by Bin Wang to generate the nbest-list
# * Run egs/hkush/s5/my/run.sh to training acoustic model and generate lattice
# * Run egs/hkush/s5/my/run_nbest.sh to generate the nbest list

import os
import numpy as np
import shutil
import re

from base import wblib as wb
from base import reader
from base import log
from trf.common import net

nbest_files_name = ('words_text',  # n-best list
                    'test_filt.txt',  # correct text
                    'acwt',  # acoustic weight
                    'lmwt.lmonly',  # language model weight
                    'lmwt.nolm')  # graph score

# local files
local_dir = '.././data/'
train = local_dir + 'train'
valid = local_dir + 'dev'
test = local_dir + 'dev'
nbests = [os.path.join(local_dir + 'nbest', x) for x in nbest_files_name]


# prepare the training text
def prepare_data(skip_exist=True):
    # verify if the data exist
    is_all_exist = True
    for s in nbests + [train, valid, test]:
        if not wb.exists(s):
            is_all_exist = False
            break
    if skip_exist and is_all_exist:
        print('[%s.prepare_data] data all exist.' % __name__)
        return

    # source files
    kaldi_root = os.environ.get('KALDI_ROOT')
    if kaldi_root is None:
        raise FileNotFoundError('[%s] $KALDI_ROOT=None' % __name__)
    kaldi_nbest_dir = os.path.join(kaldi_root, 'egs/hkust/s5/exp/chain/tdnn_7h_sp_online/decode/nbest_list/list')
    kaldi_nbest_files = [os.path.join(kaldi_nbest_dir, x) for x in nbest_files_name]
    kaldi_train_text = os.path.join(kaldi_root, 'egs/hkust/s5/data/local/train/text')
    kaldi_dev_text = os.path.join(kaldi_root, 'egs/hkust/s5/data/local/dev/text')

    # verify the source files
    for s in kaldi_nbest_files + [kaldi_train_text, kaldi_dev_text]:
        if not wb.exists(s):
            info = '[%s.prepare_data] can not find file %s.\n' % (__name__, s)
            info += 'Please install Kaldi from svn //166.111.64.101/svn/Gsp/user/wangbin/work/kaldi and run ' \
                    '$KALDI_ROOT/egs/hkush/s5/my/run.sh ' \
                    'and $KALDI_ROOT/egs/hkush/s5/my/run_nbest.sh. \n' \
                    '$KALDI_ROOT=%s' % kaldi_root
            raise FileNotFoundError(info)

    # copy the nbest list
    wb.mkdir(os.path.join(local_dir + 'nbest'), is_recreate=False)
    for x, y in zip(kaldi_nbest_files, nbests):
        shutil.copyfile(x, y)

    # copy text
    def copy_text(src, dst):
        with open(src, 'rt') as fsrc, open(dst, 'wt') as fdst:
            for line in fsrc:
                a = line.split()
                fdst.write(' '.join(a[1:]) + '\n')  # remove the label

    copy_text(kaldi_train_text, train)
    copy_text(kaldi_dev_text, valid)


# def count the data
def data_info():
    # corpus
    with open(os.path.join(local_dir + 'info.txt'), 'wt') as fp:
        train_info = wb.TxtInfo(train)
        dev_info = wb.TxtInfo(valid)

        train_info.write(fp)
        fp.write('------------------ \n\n')
        dev_info.write(fp)
        oov_words = []
        oov_num = 0
        for w, count in dev_info.vocab.items():
            if w not in train_info.vocab:
                oov_words.append((w, count))
                oov_num += count
        fp.write('oov_words={}\n'.format(len(oov_words)))
        fp.write('oov_num={}\n'.format(oov_num))
        fp.write('oov_rate={:.2f}%\n'.format(100. * oov_num / dev_info.nWord))
        with open(os.path.join(local_dir, 'oov.txt'), 'wt') as foov:
            for w, count in sorted(oov_words, key=lambda x: -x[1]):
                foov.write('{}\t{}\n'.format(w, count))

        info = wb.TxtInfo(nbests[1])
        print('utterance num: %d' % info.nLine)
        fp.write('------------------ \n')
        fp.write('utterance num: %d\n' % info.nLine)


def verify_nbest():
    # using the given scores compute the WER/CER
    nbest = NBest(*nbests)
    wb.mkdir('temp', is_recreate=False)

    print('wer=', nbest.wer())
    print('lmscale=', nbest.lmscale)

    nbest.write_1best('temp/1best.txt')
    nbest.write_log('temp/log.txt')


def word_seq_to_char(wseq):
    # remove the symbols like [laughter] [NOISE]
    # transform word to characters
    char_seq = []
    for w in wseq:
        # remove the symbols
        if w[0] == '[' and w[-1] == ']':
            continue
        # split Chinese word to char, and preserve the English words
        char_seq += wb.split_to_char_ch(w)
    return char_seq


class NBest(reader.NBest):
    """
    The NBest computation for hkush task.
      *  remove the symbols like [NOISE] [LAUGHTER]...
      *  remove the <unk>
      *  transform the Chinese words to characters, and preserve the English words
    """
    def __init__(self):
        super().__init__(*nbests)

    def wer(self, lmscale=np.linspace(1, 20, 20), rm_unk=True, sentence_process_fun=None):
        def fun(wseq):
            # remove the symbols like [laughter] [NOISE]
            # transform word to characters
            char_seq = []
            for w in wseq:
                # remove the symbols
                if w[0] == '[' and w[-1] == ']':
                    continue
                # split Chinese word to char, and preserve the English words
                cs = re.split(r'([\u4e00-\u9fa5])', w)
                cs = list(filter(None, cs))
                char_seq += cs
            return char_seq

        return super().wer(lmscale, rm_unk=True, sentence_process_fun=fun)


nbest_real = NBest()


def nbest_eval(m, data, workdir, fres, res_name, rescore_fun=None):

    def rescore(seq_list):
        if rescore_fun is not None:
            return rescore_fun(seq_list)
        else:
            return m.rescore(seq_list)

    # write nbest list
    if not wb.exists(os.path.join(workdir, 'nbest.real.id')):
        nbest_real.write_nbest_list(os.path.join(workdir, 'nbest.real.id'), data)
    save_lmscore_dir = wb.mkdir(os.path.join(workdir, 'lmscores'))

    with wb.processing('rescoring_nbest_real'):
        nbest_real.lmscore = rescore(nbest_real.get_nbest_list(data))
    nbest_real.write_lmscore(os.path.join(workdir, 'nbest.lmscore'))
    nbest_real.write_lmscore(os.path.join(save_lmscore_dir, res_name + '.lmscore'))
    wer_real = nbest_real.wer()
    #nbest_real.write_log(os.path.join(workdir, 'wer.log'))
    # print('wer={} lmscale={}, acscale={}'.format(wer_real, nbest_real.lmscale, nbest_real.acscale))
    fres.Add(res_name, ['WER', 'lmscale'], [wer_real, nbest_real.lmscale])

    return wer_real


def get_word_data():
    return reader.Data().load_raw_data([train, valid, test],
                                       add_beg_token='</s>',
                                       add_end_token='</s>',
                                       add_unknwon_token='<unk>',
                                       )


def get_char_data():
    return reader.Data().load_raw_data([train, valid, test],
                                       add_beg_token='</s>',
                                       add_end_token='</s>',
                                       add_unknwon_token='<unk>',
                                       raw_data_map=word_seq_to_char
                                       )


def get_config_cnn(vocab_size, n=200):
    config = net.Config(vocab_size)
    config.embedding_dim = 256
    config.structure_type = 'cnn'
    config.cnn_filters = [(i, 128) for i in range(1, 11)]
    config.cnn_hidden = 128
    config.cnn_width = 3
    config.cnn_layers = 3
    config.cnn_activation = 'relu'
    config.cnn_skip_connection = True
    config.cnn_residual = False
    config.cnn_batch_normalize = False
    config.cnn_final_activation = None
    return config


def get_config_rnn(vocab_size, n=200):
    config = net.Config(vocab_size)
    config.embedding_dim = 200
    config.structure_type = 'rnn'
    config.rnn_type = 'blstm'
    config.rnn_hidden_size = 200
    config.rnn_hidden_layers = 1
    config.rnn_predict = True
    config.rnn_share_emb = True
    return config

def get_config_mix(vocab_size):
    config = net.Config(vocab_size)
    config.embedding_dim = 200
    config.structure_type = 'mix'

    config.cnn_filters = [(i, 128) for i in range(1, 11)]
    config.cnn_hidden = 128
    config.cnn_width = 3
    config.cnn_layers = 3
    config.cnn_activation = 'relu'
    config.cnn_skip_connection = True
    config.cnn_residual = False
    config.cnn_batch_normalize = False

    config.rnn_type = 'blstm'
    config.rnn_hidden_size = 128
    config.rnn_hidden_layers = 1
    # config.rnn_predict = True
    # config.rnn_share_emb = True

    config.attention = True
    return config


def get_config_rnn_large(vocab_size):
    config = net.Config(vocab_size)
    config.embedding_dim = 500
    config.structure_type = 'rnn'
    config.rnn_type = 'blstm'
    config.rnn_hidden_size = 500
    config.rnn_hidden_layers = 2
    config.dropout = 0.5
    config.rnn_predict = True
    config.rnn_share_emb = True
    return config


if __name__ == '__main__':
    prepare_data(skip_exist=False)
    data_info()
    # verify_nbest()

