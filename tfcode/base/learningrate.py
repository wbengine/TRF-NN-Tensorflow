import numpy as np

from base import wblib as wb


class LearningRate(object):
    def __init__(self, init, name='lr'):
        self.init = init
        self.name = name

    def get_lr(self, t, epoch=None):
        return self.init

    def __str__(self):
        return '{} = {}'.format(self.name, self.init)


class LearningRateEpochDelay(LearningRate):
    def __init__(self, init, delay=1.0, delay_when=0, per_epoch=1):
        super().__init__(init, 'epoch_delay')
        self.delay = delay
        self.delay_when = delay_when
        self.per_epoch = per_epoch

    def get_lr(self, t, epoch=0):
        return self.init * self.delay ** (max(0, (int(epoch) - self.delay_when)//self.per_epoch))

    def __str__(self):
        return '{}: init={} delay={} when={}, per_epoch={}'.format(
            self.name, self.init, self.delay, self.delay_when, self.per_epoch
        )
