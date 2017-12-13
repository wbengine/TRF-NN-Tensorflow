import tensorflow as tf
import time
import sys
import os
import numpy as np

from model import blocklm
from model import reader
from model import wblib as wb


def small_config(data):
    config = blocklm.Config()
    config.vocab_size = data.get_vocab_size()
    config.embedding_size = 200
    config.hidden_size = 400
    config.hidden_layers = 2
    config.step_size = 20
    config.batch_size = 20
    config.block_size = 5
    config.epoch_num = 50
    config.init_weight = 0.1
    config.max_grad_norm = 5
    config.dropout = 0
    config.optimize_method = 'sgd'
    config.learning_rate = 1.0
    config.lr_decay = 1.0
    config.lr_decay_when = 4
    return config


def create_name(config):
    s = 'lstm_block{}_{}x{}'.format(config.block_size, config.hidden_size, config.hidden_layers)
    if config.softmax_type != 'Softmax':
        s += '_' + config.softmax_type
    return s

def test_softmax(_):

    inputs = tf.placeholder(dtype=tf.float32, shape=[1, 1, 10])
    labels = tf.placeholder(dtype=tf.int32, shape=[1, 1])

    vocab_size = 5
    m = lstmlm.layers.ShortlistSoftmax(inputs, labels, [3, vocab_size], 1)
    # m = lstmlm.layers.AdaptiveSoftmax(inputs, labels, [10, vocab_size])

    sv = tf.train.Supervisor()
    # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        prob_sum = 0
        prob = np.zeros(vocab_size)
        for i in range(vocab_size):
            loss = session.run(m.loss, {inputs: np.ones((1, 1, 10)), labels: [[i]]})
            print('word={}\tprob={}'.format(i, loss))
            prob[i] = np.exp(-loss[0, 0])
            prob_sum += np.exp(-loss)
        print(prob_sum)
        print(prob)

        count = np.zeros(vocab_size)
        for i in range(1000):
            draw, logp = session.run(m.draw,  {inputs: np.ones((1, 1, 10)), labels: [[i]]})
            count[draw[0, 0]] += 1
        print(count / np.sum(count))


def main(_):
    data = reader.Data().load_raw_data(reader.ptb_raw_dir(),
                                       add_beg_token=None,
                                       add_end_token='</s>',
                                       add_unknwon_token='<unk>')
    nbest = reader.NBest(*reader.wsj0_nbest())
    nbest_list = data.load_data(reader.wsj0_nbest()[0], is_nbest=True)

    config = small_config(data)
    # config = medium_config(data)
    # config = large_config(data)

    work_dir = './lstm/' + create_name(config)
    wb.mkdir(work_dir, is_recreate=True)
    sys.stdout = wb.std_log(os.path.join(work_dir, 'lstm.log'))
    print(work_dir)
    wb.pprint_dict(config.__dict__)

    data.write_vocab(work_dir + '/vocab.txt')
    data.write_data(data.datas[0], work_dir + '/train.id')
    data.write_data(data.datas[1], work_dir + '/valid.id')
    data.write_data(data.datas[2], work_dir + '/test.id')
    data.write_data(nbest_list, work_dir + '/nbest.id')

    write_model = os.path.join(work_dir, 'model.ckpt')

    with tf.Graph().as_default():
        # lm = lstmlm.FastLM(config, device_list=['/gpu:0', '/gpu:0'])
        lm = blocklm.LM(config, device='/gpu:0')
        param_num = tf.add_n([tf.size(v) for v in tf.trainable_variables()])

        for v in lm.train_net.variables:
            print(v.name)

        save = tf.train.Saver()

        # used to write ppl on valid/test set
        summ_bank = blocklm.layers.SummaryScalarBank(['ppl_valid', 'ppl_test'])
        summ_var = blocklm.layers.SummaryVariables()

        sv = tf.train.Supervisor(logdir=os.path.join(work_dir, 'logs'),
                                 summary_op=None, global_step=lm.global_step())
        sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:

            print('param_num={:,}'.format(session.run(param_num)))

            lm.train(sv, session,
                     data.datas[0],
                     data.datas[1],
                     data.datas[2])

            save.save(session, write_model)

if __name__ == '__main__':
    tf.app.run(main=main)
