import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from trf.sa import trf as trf

from eval import Operation, get_config_rnn, get_config_cnn, nbest_eval_chr


def get_config(data):
    config = trf.Config(data)
    config.feat_config = None
    config.net_config.update(get_config_rnn(data.get_vocab_size()))

    config.chain_num = 100
    config.multiple_trial = 4
    config.jump_width = 3
    config.sample_sub = 3

    config.lr_net = lr.LearningRateTime(1.0, 0.4, tc=1e2)
    config.lr_logz = lr.LearningRateTime(1, 0.2)
    config.lr_sampler = lr.LearningRateEpochDelay(1, 0.5, delay_when=5)
    config.opt_feat_method = 'momentum'
    config.opt_net_method = 'momentum'
    config.opt_logz_method = 'sgd'
    config.max_epoch = 1000
    config.train_batch_size = 1000
    config.sample_batch_size = 100
    # config.init_logz = np.zeros_like(config.init_logz)

    # config.write_dbg = True

    return config


class OpsForSampler(wb.Operation):
    def __init__(self, sampler, data, logdir):
        super().__init__()
        self.sampler = sampler
        self.data = data
        self.logdir = logdir

    def perform(self, step, epoch):
        nbest_eval_chr(self.sampler, self.data, self.logdir,
                       res_file=os.path.join(self.logdir, 'sampler_wer_per_epoch.log'),
                       res_name='epoch%.2f' % epoch,
                       rescore_fun=lambda x: -self.sampler.get_log_probs(x)
                       )


class OpsForTRF(Operation):
    def __init__(self, m):
        super().__init__(m)
        self.perform_next_epoch = 1.0
        self.perform_per_epoch = 1.0

    def perform(self, step, epoch):

        if 'sampler' in self.m.__dict__:
            nbest_eval_chr(self.m, self.m.data, self.m.logdir,
                           res_file=os.path.join(self.m.logdir, 'sampler_wer_per_epoch.log'),
                           res_name='epoch%.2f' % epoch,
                           rescore_fun=lambda x: -self.m.sampler.get_log_probs(x)
                           )

        super().perform(step, epoch)


def main():
    with open('data.info') as f:
        info = json.load(f)

    data = reader.Data().load_raw_data(file_list=info['hkust_train_chr'] +
                                                 info['hkust_valid_chr'] +
                                                 info['hkust_valid_chr'],
                                       add_beg_token='<s>',
                                       add_end_token='</s>',
                                       add_unknwon_token='<unk>')

    config = get_config(data)
    logdir = wb.mklogdir('trf_sa/' + str(config), is_recreate=True)
    config.print()

    config.net_config.load_embedding_path = os.path.join(logdir, 'word_emb.txt')
    if config.net_config.load_embedding_path is not None:
        # get embedding vectors
        data.word2vec(config.net_config.load_embedding_path, config.net_config.embedding_dim, cnum=0)

    data.write_vocab(os.path.join(logdir, 'vocab.chr'))
    data.write_data(data.datas[0], os.path.join(logdir, 'train.id'))
    data.write_data(data.datas[1], os.path.join(logdir, 'valid.id'))

    m = trf.TRF(config, data, logdir, device='/gpu:0')

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.train(0.1, operation=OpsForTRF(m))


if __name__ == '__main__':
    main()
