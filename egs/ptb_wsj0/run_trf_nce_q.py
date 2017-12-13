import tensorflow as tf
import os
import sys
import numpy as np

from model import wblib as wb
from model import reader
from model import trfbase
from model import trfnce
from model import lstmlm
import run_lstmlm_withBegToken


def create_name(config):
    s = str(config) + '_withq'
    if config.dropout > 0:
        s += '_d{}'.format(config.dropout)
    s += '_logz{}'.format(config.init_zeta)
    return s


def main(_):
    data = reader.Data().load_raw_data(reader.ptb_raw_dir(),
                                       add_beg_token='<s>', add_end_token='</s>',
                                       add_unknwon_token='<unk>')
    nbest = reader.NBest(*reader.wsj0_nbest())
    nbest_list = data.load_data(nbest.nbest, is_nbest=True)

    config = trfnce.Config(data)
    config.structure_type = 'cnn'
    config.embedding_dim = 256
    config.cnn_filters = [(i, 128) for i in range(1, 11)]
    config.cnn_hidden = 128
    config.cnn_layers = 3
    config.cnn_skip_connection = True
    config.cnn_activation = 'tanh'
    config.batch_size = 50
    config.noise_factor = 20
    config.noise_sampler = 2
    config.init_weight = 0.1
    config.lr_param = trfbase.LearningRateTime(1e-3)
    config.max_epoch = 10
    config.global_normalized = True
    config.update_zeta = False
    config.init_zeta = 10
    config.dropout = 0
    config.write_dbg = False
    config.pprint()

    name = create_name(config)
    logdir = 'trf_nce/' + name
    wb.mkdir(logdir, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(logdir, 'trf.log'))
    print(logdir)

    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')
    data.write_data(nbest_list, logdir + '/nbest.id')

    # wb.rmdir(logdirs)
    with tf.Graph().as_default():

        m = trfnce.TRF(config, data,
                       logdir=logdir, device='/gpu:1',
                       q_model=lstmlm.LM(run_lstmlm_withBegToken.small_config(data), device='/gpu:1')
                       )

        sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'),
                                 global_step=m.train_net.global_step)
        sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:
            m.set_session(session)

            print('load lstmlm for noise generator')
            m.q_model.restore(session,
                              './lstm/' + run_lstmlm_withBegToken.create_name(m.q_model.config) + '/model.ckpt')

            m.train(sv, session,
                    print_per_epoch=0.1,
                    nbest=nbest,
                    nbest_list=nbest_list
                    )

if __name__ == '__main__':
    tf.app.run(main=main)
