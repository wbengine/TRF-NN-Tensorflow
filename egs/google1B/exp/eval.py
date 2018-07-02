import os
import json
from base import *
from trf.common import net
from hrf.crf import DefaultOps

# with open('data.info') as f:
#     info = json.load(f)


# nbest_valid = NBest(*info['nbest_valid_chr'])
nbest_test = reader.NBest(*reader.wsj0_nbest())


def nbest_eval(m, data, workdir, res_file, res_name, rescore_fun=None):

    def rescore(seq_list):
        if rescore_fun is not None:
            return rescore_fun(seq_list)
        else:
            return m.rescore(seq_list)

    save_lmscore_dir = wb.mkdir(os.path.join(workdir, 'lmscores'))
    fres = wb.FRes(res_file)

    wers = []
    lmscales = []
    opt_lmscale = None
    for nbest, flag in zip([nbest_test], ['test']):
        # write nbest list
        if wb.exists(os.path.join(workdir, 'nbest.%s.id' % flag)):
            nbest.write_nbest_list(os.path.join(workdir, 'nbest.%s.id' % flag), data)

        # rescore
        with wb.processing('rescore_nbest'):
            nbest.lmscore = rescore(nbest.get_nbest_list(data))

        nbest.write_lmscore(os.path.join(workdir, 'nbest.%s.lmscore' % flag))
        nbest.write_lmscore(os.path.join(save_lmscore_dir, res_name + '.%s.lmscore' % flag))

        if opt_lmscale is None:
            wer = nbest.wer()
            opt_lmscale = nbest.lmscale
        else:
            wer = nbest.wer(lmscale=[opt_lmscale])
        wers.append(wer)
        lmscales.append(nbest.lmscale)
        nbest.write_log(os.path.join(workdir, 'wer.%s.log' % flag))

        fres.Add(res_name, ['WER-%s' % flag, 'lmscale-%s' % flag], [wer, nbest.lmscale])

    print('{} WER-test={:.3f} lmscale-test={:.3f}'.format(
        res_name, wers[0], lmscales[0],
    ))

    return wers, lmscales


def nbest_eval_lmscore(lmscore_list, res_file, res_name, head):
    fres = wb.FRes(res_file)
    print('cmp wer: ', res_name)

    wers = []
    lmscales = []
    opt_lmscale = None
    for nbest, flag, lmscore in zip([nbest_test], ['test'], [lmscore_list]):
        nbest.lmscore = wb.LoadScore(lmscore) if isinstance(lmscore, str) else lmscore

        if opt_lmscale is None:
            wer = nbest.wer()
            opt_lmscale = nbest.lmscale
        else:
            wer = nbest.wer(lmscale=[opt_lmscale])
        wers.append(wer)
        lmscales.append(nbest.lmscale)
        fres.Add(res_name, ['[%s]WER-%s' % (head, flag), '[%s]lmscale-%s' % (head, flag)], [wer, nbest.lmscale])

    print('{}{} WER-test={:.3f} lmscale-test={:.3f}'.format(
        res_name, head, wers[0],  lmscales[0]
    ))

    return wers, lmscales


class OperationLSTM(wb.Operation):
    def __init__(self, m, data, logdir, rescore_fun):
        super().__init__()

        self.data = data
        self.logdir = logdir
        self.rescore_fun = rescore_fun
        self.m = m
        self.perform_next_epoch = 0.0
        self.perform_per_epoch = 1.0

    def perform(self, step, epoch):
        nbest_eval(self.m, self.data, self.logdir,
                   res_file=os.path.join(self.logdir, 'wer_per_epoch.log'),
                   res_name='epoch%.2f' % epoch,
                   rescore_fun=self.rescore_fun
                  )

class Operation(DefaultOps):
    def __init__(self, m, data=None):
        self.data = data
        if data is not None:
            super().__init__(m, data.datas[-1])

        self.m = m
        self.perform_next_epoch = 0.0
        self.perform_per_epoch = 1.0

    def perform(self, step, epoch):

        if self.data is not None:
            super().perform(step, epoch)

        try:
            self.m.update_global_norm()
        except AttributeError:
            print('model no function: update_global_norm()')

        nbest_eval(self.m, self.m.data, self.m.logdir,
                   res_file=os.path.join(self.m.logdir, 'wer_per_epoch.log'),
                   res_name='epoch%.2f' % epoch
                  )


def get_config_rnn(vocab_size, n=200):
    config = net.Config(vocab_size)
    config.embedding_dim = n
    config.structure_type = 'rnn'
    config.rnn_type = 'blstm'
    config.rnn_hidden_size = n
    config.rnn_hidden_layers = 1
    config.rnn_predict = True
    config.rnn_share_emb = True
    return config


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