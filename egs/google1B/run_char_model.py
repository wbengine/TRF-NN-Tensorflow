import tensorflow as tf
import sys
import os
import corpus

from model import reader
from model import trf
from model import wblib as wb


def create_name(config):
    s = 'trf'
    if config.feat_type_file is not None:
        s += '_{}'.format(os.path.splitext(os.path.split(config.feat_type_file)[-1])[0])
    s += '_e{}_cnn{}_{}'.format(config.embedding_dim,
                                '' if config.cnn_final == 'linear' else config.cnn_final,
                                config.cnn_activation)
    if config.load_embedding_path is not None:
        s += '_loademb'
    if config.cnn_shared_over_layers:
        s += '_shared'
    if config.cnn_skip_connection:
        s += '_skipconn'
    if config.cnn_filters is not None:
        s += '({}to{}x{})'.format(config.cnn_filters[0][0], config.cnn_filters[-1][0],
                                  config.cnn_filters[0][1])
    if config.cnn_layers > 0:
        s += '_({}x{})x{}'.format(config.cnn_width, config.cnn_hidden, config.cnn_layers)

    s += '_{}'.format(config.opt_method)
    if config.feat_type_file is not None and config.L2_reg > 0:
        s += '_reg{:.0e}'.format(config.L2_reg)
    return s


def main(_):
    data = reader.Data().load_raw_data(corpus.char_raw_dir(),
                                       add_beg_token='<s>', add_end_token='</s>',
                                       add_unknwon_token=None,
                                       max_length=1000)
    nbest = reader.NBest(*reader.wsj0_nbest())
    print(nbest.wer())

    config = trf.trfbase.Config(data)
    config.embedding_dim = 12
    config.cnn_filters = [(i, 12) for i in range(1, 11)]
    config.cnn_layers = 3
    config.cnn_hidden = 12
    config.cnn_shared_over_layers = False
    config.cnn_residual = True
    config.cnn_skip_connection = True
    config.max_epoch = 1000
    config.sample_sub = 100
    config.jump_width = 10
    config.init_weight = 0.1
    config.opt_method = 'adam'
    config.lr_cnn = trf.trfbase.LearningRateTime(1, 1.5, tc=1e4)
    config.lr_zeta = trf.trfbase.LearningRateTime(1.0, 0.2)
    config.load_embedding_path = './embedding/ptb_{}x{}.emb'.format(config.vocab_size, config.embedding_dim)
    config.auxiliary_hidden = 12
    config.auxiliary_lr = 1.0

    name = create_name(config)
    logdir = name
    wb.mkdir(logdir, is_recreate=True)
    sys.stdout = wb.std_log(logdir + '/trf.log')
    print(logdir)
    config.pprint()

    # prapare embedding
    if wb.is_linux() and config.load_embedding_path is not None or \
            (config.feat_type_file and config.feat_cluster > 0):
        if config.load_embedding_path is None:
            fvectors = './embedding/ptb_{}x{}.emb'.format(config.vocab_size, config.embedding_dim)
        else:
            fvectors = config.load_embedding_path
        data.word2vec(fvectors, dim=config.embedding_dim, cnum=config.feat_cluster)
    else:
        config.load_embedding_path = None

    # write data
    data.write_vocab(logdir + '/vocab.txt')
    data.write_data(data.datas[0], logdir + '/train.id')
    data.write_data(data.datas[1], logdir + '/valid.id')
    data.write_data(data.datas[2], logdir + '/test.id')

    nbest_char_txt = logdir + '/nbest.char.txt'
    corpus.word_text_to_char_text(reader.wsj0_nbest()[0], nbest_char_txt, is_nbest=True)
    nbest_list = data.load_data(nbest_char_txt, is_nbest=False)
    data.write_data(nbest_list, logdir + '/nbest.id')

    with tf.Graph().as_default():
        m = trf.TRF(config, data, logdir=logdir, device='/gpu:2', simulater_device='/gpu:1')

        sv = tf.train.Supervisor(logdir=logdir + '/logs', summary_op=None, global_step=m._global_step)
        # sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs

        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        with sv.managed_session(config=session_config) as session:

            # s = ['it was not black monday', 'we did n\'t even get a chance']
            # eval_list = data.load_data([[data.beg_token_str] + w.split() + [data.end_token_str] for w in s])
            # print(eval_list)

            # import sampling as sp
            # x_batch = [x for x in sp.SeqIter(3, config.vocab_size,
            #                                  beg_token=config.beg_token,
            #                                  end_token=config.end_token)]
            # logprobs = m.get_log_probs(x_batch, False)
            # logz = sp.log_sum(logprobs)
            # print(logprobs)
            # print(logz)

            m.train(session, sv,
                    print_per_epoch=0.1,
                    nbest=nbest,
                    nbest_list=nbest_list)


if __name__ == '__main__':
    tf.app.run(main=main)
