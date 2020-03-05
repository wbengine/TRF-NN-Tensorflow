import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from trf.isample import trf_nce as trf
from hrf.crf import DefaultOps

# from trf.nce import trf

#
# from eval import Operation, nbest_eval_chr, get_config_rnn, get_config_cnn

nbest_path = '/Users/lls/tmp/tlt2020_local_eval/nbest/nbest'
refer_path = '/Users/lls/tmp/tlt2020_local_eval/nbest/ref'
acwt_path = '/Users/lls/tmp/tlt2020_local_eval/nbest/acwt'
grwt_path = '/Users/lls/tmp/tlt2020_local_eval/nbest/grwt'
lmwt_path = '/Users/lls/tmp/tlt2020_local_eval/nbest/lmwt'
nbest_valid = reader.NBest(nbest_path, refer_path, acwt_path, lmwt_path, grwt_path)

train_text = 'data/TLT16W17train.trn.txt'
dev_text = 'data/TLT2017dev.trn.txt'
test_text = 'data/TLT2017dev.trn.txt'

out_dir = 'trf_nce'


def get_config_rnn(vocab_size, n=200):
    from trf.common import net
    config = net.Config(vocab_size)
    config.embedding_dim = n
    config.structure_type = 'rnn'
    config.rnn_type = 'blstm'
    config.rnn_hidden_size = n
    config.rnn_hidden_layers = 1
    config.rnn_predict = True
    config.rnn_share_emb = True
    return config


def get_config(data):
    config = trf.Config(data)
    # config.pi_0 = data.get_pi0(config.pi_true)
    # config.pi_true = config.pi_0
    config.norm_config = 'linear'
    config.batch_size = 100
    config.noise_factor = 4
    config.data_factor = 0
    config.train_add_noise = False

    # config.noise_sampler = '2gram'

    # config.lr_feat = lr.LearningRateTime(1e-4)
    config.lr_net = lr.LearningRateTime(1e-3)  # lr.LearningRateTime(1, 0.5, tc=1e3)
    config.lr_logz = lr.LearningRateTime(1e-2)
    config.lr_sampler = lr.LearningRateEpochDelay(0.1)
    config.opt_feat_method = 'adam'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'adam'
    config.max_epoch = 1000
    # sampler
    # config.sampler_config.hidden_layers = 2
    # config.load_sampler = 'sampler/lstm_e200_h200x2/sampler.ckpt'
    # config.fix_sampler = True

    config.init_logz = config.get_initial_logz()
    config.init_global_logz = 0

    config.feat_config = None

    # net config
    config.net_config.update(get_config_rnn(config.vocab_size))
    # config.net_config.l2_reg = 1e-4
    # wb.mkdir('word_emb')
    # config.net_config.load_embedding_path = 'word_emb/ptb_d{}.emb'.format(config.net_config.embedding_dim)

    config.write_dbg = False
    config.add_sampler_as_prior = False

    return config


def nbest_eval_chr(m, data, workdir, res_file, res_name, rescore_fun=None):
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
    for nbest, flag in zip([nbest_valid], ['valid']):
        # write nbest list
        if wb.exists(os.path.join(workdir, 'nbest.%s.id' % flag)):
            nbest.write_nbest_list(os.path.join(workdir, 'nbest.%s.id' % flag), data)

        # rescore
        with wb.processing('rescore_nbest'):
            nbest.lmscore = rescore(nbest.get_nbest_list(data))

        nbest.write_lmscore(os.path.join(workdir, 'nbest.%s.lmscore' % flag))
        nbest.write_lmscore(os.path.join(save_lmscore_dir, res_name + '.%s.lmscore' % flag))

        if opt_lmscale is None:
            wer = nbest.wer(lmscale=np.linspace(0, 20, 21))
            opt_lmscale = nbest.lmscale
        else:
            wer = nbest.wer(lmscale=[opt_lmscale])
        wers.append(wer)
        lmscales.append(nbest.lmscale)
        nbest.write_log(os.path.join(workdir, 'wer.%s.log' % flag))

        fres.Add(res_name, ['WER=%s' % flag, 'lmscale-%s' % flag], [wer, nbest.lmscale])

    print('{} WER-valid={:.3f} lmscale-valid={:.3f} '.format(
        res_name, wers[0], lmscales[0],
    ))

    return wers, lmscales


class Ops(wb.Operation):
    def __init__(self, m, only_sampler=False):
        super().__init__()
        self.only_sampler = only_sampler
        self.m = m

        self.perform_next_epoch = 0

    def perform(self, step, epoch):

        if 'sampler' in self.m.__dict__:
            nbest_eval_chr(self.m, self.m.data, self.m.logdir,
                           res_file=os.path.join(self.m.logdir, 'sampler_wer_per_epoch.log'),
                           res_name='sampler_epoch%.2f' % epoch,
                           rescore_fun=lambda x: -self.m.sampler.get_log_probs(x)
                           )

        if not self.only_sampler:
            nbest_eval_chr(self.m, self.m.data, self.m.logdir,
                           res_file=os.path.join(self.m.logdir, 'wer_per_epoch.log'),
                           res_name='epoch%.2f' % epoch
                           )


def main():
    data = reader.Data().load_raw_data(file_list=[train_text, dev_text, test_text],
                                       add_beg_token='</s>',
                                       add_end_token='</s>',
                                       add_unknwon_token='<unk>')

    config = get_config(data)
    logdir = wb.mklogdir(out_dir + '/' + str(config), is_recreate=True, force=True)
    config.print()

    # config.word_config.load_embedding_path = os.path.join(logdir, 'word_emb.txt')
    # if config.word_config.load_embedding_path is not None:
    #     # get embedding vectors
    #     data.create_data().word2vec(config.word_config.load_embedding_path, config.word_config.embedding_dim, cnum=0)

    data.write_vocab(os.path.join(logdir, 'vocab.chr'))
    data.write_data(data.datas[0], os.path.join(logdir, 'train.id'))
    data.write_data(data.datas[1], os.path.join(logdir, 'valid.id'))

    m = trf.TRF(config, data, logdir, device='/gpu:1')
    print('sampler' in m.__dict__)

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.train(0.1, Ops(m))

            # write_dir, write_name = os.path.split(config.load_sampler)
            # m.pretrain_sampler(max_epoch=20, print_per_epoch=0.1,
            #                    operation=Ops(m, only_sampler=True),
            #                    write_sampler=os.path.join(wb.mkdir(write_dir), write_name))


if __name__ == '__main__':
    main()
