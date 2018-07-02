import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from trf.isample import trf_nce as trf
# from trf.nce import trf

from eval import Operation, nbest_eval, nbest_eval_lmscore, get_config_rnn, get_config_cnn


def get_config(data):
    config = trf.Config(data)
    # config.pi_0 = data.get_pi0(config.pi_true)
    # config.pi_true = config.pi_0
    config.norm_config = 'linear'
    config.batch_size = 200
    config.noise_factor = 4
    config.data_factor = 0.5

    # config.lr_feat = lr.LearningRateTime(1e-4)
    config.lr_net = lr.LearningRateTime(1e-3)  # lr.LearningRateTime(1, 0.5, tc=1e3)
    config.lr_logz = lr.LearningRateTime(1e-2)
    config.lr_sampler = lr.LearningRateEpochDelay(0.1)
    config.opt_feat_method = 'adam'
    config.opt_net_method = 'adam'
    config.opt_logz_method = 'adam'
    config.max_epoch = 1000
    # sampler
    config.sampler_config.vocab_size = 10000
    config.sampler_config.max_batch_per_core = 256

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


class Ops(Operation):
    def __init__(self, m):
        super().__init__(m)
        self.perform_next_epoch = 0
        self.perform_per_epoch = 0.1

        self.ngram_lmscore = 'ngramlm/t1_KN3_002/nbest.lmscore'

    def perform(self, step, epoch):

        if 'sampler' in self.m.__dict__:
            nbest_eval(self.m, self.m.data, self.m.logdir,
                       res_file=os.path.join(self.m.logdir, 'sampler_wer_per_epoch.log'),
                       res_name='epoch%.2f' % epoch,
                       rescore_fun=lambda x: -self.m.sampler.get_log_probs(x)
                       )

        super().perform(step, epoch)

        # model combine
        cmb_socre = 0.5 * wb.LoadScore(self.ngram_lmscore) + \
                    0.5 * wb.LoadScore(os.path.join(self.m.logdir, 'nbest.test.lmscore'))

        nbest_eval_lmscore(cmb_socre,
                           res_file=os.path.join(self.m.logdir, 'sampler_wer_per_epoch.log'),
                           res_name='epoch%.2f' % epoch,
                           head='+KN3')


def main():
    with open('../data.info') as f:
        data_info = json.load(f)

    train_files = 100
    data = reader.LargeData().dynamicly_load_raw_data(sorted_vocab_file=None,
                                                      train_list=data_info['train_all'][0: train_files],
                                                      valid_file=data_info['valid'],
                                                      test_file=data_info['test'],
                                                      max_length=60,
                                                      add_beg_token='<s>',
                                                      add_end_token='</s>',
                                                      add_unknwon_token='<unk>',
                                                      vocab_max_size=None,
                                                      vocab_cutoff=3)

    config = get_config(data)
    logdir = wb.mklogdir('trf_t%d_nce/' % train_files + str(config), is_recreate=True)
    config.print()

    # config.word_config.load_embedding_path = os.path.join(logdir, 'word_emb.txt')
    # if config.word_config.load_embedding_path is not None:
    #     # get embedding vectors
    #     data.create_data().word2vec(config.word_config.load_embedding_path, config.word_config.embedding_dim, cnum=0)

    data.write_vocab(os.path.join(logdir, 'vocab.chr'))
    data.write_data(data.datas[0], os.path.join(logdir, 'train.id'))
    data.write_data(data.datas[1], os.path.join(logdir, 'valid.id'))

    m = trf.TRF(config, data, logdir, device=['/gpu:0', '/gpu:1'])
    print('sampler' in m.__dict__)

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.train(0.1, Ops(m))


if __name__ == '__main__':
    main()
