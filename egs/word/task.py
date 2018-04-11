import numpy as np
import os
import time

from base import *
from trf.common import net


train = 'data/train.words'
valid = 'data/valid.words'
test = 'data/test.words'

# train_sents = 1625584
# train_words = 37243300
# train_vocab = 4983
# valid_sents = 3280
# valid_words = 54239
# valid_vocab = 1596

nbest_dir = 'data/nbest.words'
trans_dir = 'data/transcript.words'


def get_data():
    return train, valid, test


def get_nbest():
    return nbest_dir, trans_dir


def get_config_cnn(vocab_size, n=32):
    config = net.Config(vocab_size)
    config.embedding_dim = n
    config.structure_type = 'cnn'
    config.cnn_filters = [(i, n) for i in range(1, 6)]
    config.cnn_hidden = n
    config.cnn_width = 3
    config.cnn_layers = 3
    config.cnn_skip_connection = True
    config.cnn_residual = False
    config.cnn_activation = 'relu'
    config.cnn_batch_normalize = False
    return config


def get_config_rnn(vocab_size, n=16):
    config = net.Config(vocab_size)
    config.embedding_dim = n
    config.structure_type = 'rnn'
    config.rnn_hidden_layers = 1
    config.rnn_hidden_size = n
    config.rnn_predict = True
    config.rnn_type = 'blstm'
    return config










