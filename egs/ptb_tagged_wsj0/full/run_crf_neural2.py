#############################
# using the new CRF code in semi/
# which is much faster
#############################

import json
import os
import tensorflow as tf
from tqdm import tqdm

from base import *
from semi import crf
from hrf import alg as alg_np


def main():
    with open('data.info') as f:
        data_info = json.load(f)

    data = seq.Data(vocab_files=data_info['vocab'],
                    train_list=data_info['train'],
                    valid_list=data_info['valid'],
                    test_list=data_info['test']
                    )

    config = crf.Config(data)

    # features
    config.mix_config.c2w_type = 'rnn'
    config.mix_config.chr_embedding_size = 100
    config.mix_config.c2w_rnn_size = 100
    config.mix_config.opt_method = 'adam'
    config.mix_config.dropout = 0.5

    config.train_batch_size = 20

    config.lr_mix = lr.LearningRateEpochDelay2(1e-3, delay=0.05)
    config.max_epoch = 100
    config.print()

    logdir = wb.mklogdir('crf2/' + str(config) + '_%s' % config.mix_config.opt_method, is_recreate=True)

    data.vocabs[0].write(os.path.join(logdir, 'vocab.chr'))
    data.vocabs[1].write(os.path.join(logdir, 'vocab.tag'))
    data.write_file(data.datas[1], os.path.join(logdir, 'valid.id'))
    data.write_file(data.datas[2], os.path.join(logdir, 'test.id'))

    m = crf.CRF(config, data, logdir, device='/gpu:1')

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            ops = crf.DefaultOps(m, data.datas[1], data.datas[2])
            ops.perform_next_epoch = 0
            m.train(0.2, ops)
            # m.restore()
            #
            # # print(m.eval(data.datas[1]))
            # m.debug_get_logpx()
            #
            # seq_list = data.datas[1][0:2]
            # for s in seq_list:
            #     print(s)
            # inputs, labels, lengths = crf.seq_list_package(seq_list)
            # phi = m.phi(inputs, labels, lengths)
            # logz = m.logz(inputs, lengths)
            # print('phi=', phi)
            # print('logz=', logz)
            #
            # trans_mat = session.run(m.phi_mix.edge_matrix)
            # emiss_mats = m.phi_mix.run_outputs(session, inputs, lengths)
            # logz = []
            # for emat, n in zip(emiss_mats, lengths):
            #     fb = alg_np.ForwardBackward(trans_mat, trans_mat, emat[0:n],
            #                                 [data.get_beg_tokens()[1]],
            #                                 [data.get_end_tokens()[1]])
            #     logz.append(fb.logsum())
            # print(logz)


if __name__ == '__main__':
    main()
