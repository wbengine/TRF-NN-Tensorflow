import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from trf.fDiv import trf
# from trf.nce import trf

from eval import Operation, nbest_eval_chr, get_config_rnn, get_config_cnn


def get_config(data):
    config = trf.Config(data)
    config.batch_size = 100
    config.sample_batch_size = 400

    config.lr_net = lr.LearningRateTime(1e-3)  # lr.LearningRateTime(1, 0.5, tc=1e3)
    # config.lr_logz = lr.LearningRateTime(1e-3)
    config.lr_sampler = lr.LearningRateEpochDelay(0.1)
    config.max_epoch = 1000

    # sampler
    config.sampler_config = trf.sampler.LSTMLen.Config(config.vocab_size, 200, 1)
    # config.load_sampler = 'sampler/lstm_e200_h200x2/sampler.ckpt'
    # config.fix_sampler = True

    # net config
    config.net_config.update(get_config_cnn(config.vocab_size))
    config.net_config.fdiv_type = 'nce4'
    config.net_config.opt_method = 'adam'
    config.net_config.init_logz_a = 0.
    config.net_config.init_logz_b = 0

    config.write_dbg = True
    return config


class Ops(Operation):
    def __init__(self, m, only_sampler=False):
        super().__init__(m)
        self.only_sampler = only_sampler

        self.perform_next_epoch = 1.0

    def perform(self, step, epoch):

        if 'sampler' in self.m.__dict__:
            nbest_eval_chr(self.m, self.m.data, self.m.logdir,
                           res_file=os.path.join(self.m.logdir, 'sampler_wer_per_epoch.log'),
                           res_name='epoch%.2f' % epoch,
                           rescore_fun=lambda x: -self.m.sampler.get_log_probs(x)
                           )

        if not self.only_sampler:
            super().perform(step, epoch)


def main():
    with open('data.info') as f:
        info = json.load(f)

    data = reader.Data().load_raw_data(file_list=info['hkust_train_chr'] +
                                                 info['hkust_valid_chr'] +
                                                 info['hkust_valid_chr'],
                                       add_beg_token='</s>',
                                       add_end_token='</s>',
                                       add_unknwon_token='<unk>')

    config = get_config(data)
    logdir = wb.mklogdir('trf_fdiv/' + str(config), is_recreate=True)
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
