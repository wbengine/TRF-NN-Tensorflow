import tensorflow as tf
import sys
import os
import numpy as np
import time

from model import lstmlm
from model import blocklm
from model import reader
from model import wblib as wb


def main(_):
    # [data]
    data = reader.Data().load_raw_data(reader.ptb_raw_dir(), add_beg_token=None, add_end_token='</s>')

    config = blocklm.Config()
    config.vocab_size = data.get_vocab_size()
    config.block_size = 5
    config.hidden_layers = 1

    m = lstmlm.LM(config)

    m2 = blocklm.LM(config)
    for v in m2.train_net.variables:
        print(v.name)

    wb.pprint_dict(config.__dict__)

    recoder = wb.clock()

    sv = tf.train.Supervisor()
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        batch = m.sample_net.config.batch_size
        length = 1000

        initial_seqs = np.random.choice(config.vocab_size, size=(batch, 1))

        with recoder.recode('sample_block'):
            final_saqs = m2.simulate(session, initial_seqs, length, True)
            # for i in range(length // config.block_size):
            #     append_seqs, _ = m2.sample_net.run_predict(session, initial_seqs[:, -1:], m2.sample_net.draw)

        with recoder.recode('sample_lstm'):
            final_saqs = m.simulate(session, initial_seqs, length, True)

            # for i in range(length):
            #     append_seqs, _ = m.sample_net.run_predict(session, initial_seqs[:, -1:], m.sample_net.draw)



        # with recoder.recode('sample'):
        #     m.sample_net.set_zero_state(session)
        #     for i in range(length):
        #         append_seqs, _ = m.sample_net.run_predict(session, initial_seqs[:, -1:], m.sample_net.draw)
        #         initial_seqs = np.concatenate([initial_seqs, append_seqs], axis=-1)
        #     # print(initial_seqs)
        #
        # with recoder.recode('probs'):
        #     m.sample_net.set_zero_state(session)
        #     for i in range(length):
        #         probs = m.sample_net.run_predict(session, initial_seqs[:, i:i+1], [m.sample_net.softmax.probs])
        #     # print(initial_seqs)
        #
        # with recoder.recode('condition'):
        #     m.sample_net.set_zero_state(session)
        #     for i in range(length):
        #         m.sample_net.run(session, initial_seqs[:, i:i+1], initial_seqs[:, i+1:i+2], [m.sample_net.cost])
        #     # print(initial_seqs)

        for key, t in sorted(recoder.items(), key=lambda x:x[0]):
            print('{}={:.2f}'.format(key, t * 60))


if __name__ == '__main__':
    tf.app.run(main=main)
