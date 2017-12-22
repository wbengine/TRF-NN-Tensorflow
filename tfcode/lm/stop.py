import numpy as np

from base import *


class EarlyStop(object):
    def __init__(self, init_lr, min_lr, delay_min_rate, nondelay_max_times=2):
        """
        Args:
            init_lr: the initial learning rate
            min_lr: if the lr less than min_lr, then stop iteration
            delay_min_rate: if delay rate less than this rate, then half the learning rate
            nondelay_max_times: if the value ascends this times successively, then stop the learning rate
        """
        self.lr = init_lr
        self.min_lr = min_lr
        self.delay_min_rate = delay_min_rate
        self.nondelay_max_times = nondelay_max_times

        self.nondelay_cur_times = 0
        self.last_value = None

    def verify(self, value):
        """

        Args:
            value: the new value

        Returns:
            None: stop the iteration
            or
            lr: the new learning rate
        """
        if self.last_value is None:
            self.last_value = value
            return self.lr

        if value >= self.last_value:
            self.nondelay_cur_times += 1
            rate = 1
        else:
            self.nondelay_cur_times = 0
            rate = (self.last_value - value) / self.last_value

        self.last_value = value

        print('[{}.{}] delay_rate={:.3f} ascent_times={}'.format(
            __name__, self.__class__.__name__, rate, self.nondelay_cur_times
        ))

        if self.nondelay_cur_times >= self.nondelay_max_times:
            return None

        if rate < self.delay_min_rate or self.nondelay_cur_times > 0:
            self.lr /= 2
            if self.lr < self.min_lr:
                return None
        return self.lr



