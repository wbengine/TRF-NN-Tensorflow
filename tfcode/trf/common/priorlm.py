import tensorflow as tf
import numpy as np
import os

from base import *
from lm import *


class EmptyLM(object):
    def initialize(self):
        pass

    def get_log_probs(self, seq_list):
        return np.zeros(len(seq_list))


class LSTMLM(EmptyLM):
    def __init__(self, load_name, device='/gpu:0'):
        print('[{}.LSTMLM]: load lstm lm from {} to device {}'.format(__name__, load_name, device))
        self.load_name = load_name
        self.lm = lstmlm.load(load_name, device)

    def initialize(self):
        print('[{}.LSTMLM]: restore lstm lm from {}'.format(__name__, self.lm.default_path))
        self.lm.restore(tf.get_default_session())

    def get_log_probs(self, seq_list):
        return -self.lm.rescore(tf.get_default_session(), seq_list,
                                reset_state_for_sentence=True,
                                pad_end_token_to_head=False)


