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


class LearningRateEpochDelay2(LearningRate):
    def __init__(self, init, delay=1.0, delay_when=0, per_epoch=1):
        super().__init__(init, 'epoch_delay2')
        self.delay = delay
        self.delay_when = delay_when
        self.per_epoch = per_epoch

    def get_lr(self, t, epoch=0):
        return self.init / (1 + self.delay * (epoch - self.delay_when) / self.per_epoch)

    def __str__(self):
        return '{}: init={} delay={} when={}, per_epoch={}'.format(
            self.name, self.init, self.delay, self.delay_when, self.per_epoch
        )


class LearningRateTime(LearningRate):
    def __init__(self, a=1., beta=0., t0=None, lr_min=0., lr_max=1000., tc=0):
        super().__init__('time')
        self.a = a
        self.beta = beta
        self.t0 = t0
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.tc = tc

    def get_lr(self, t, epoch=0):
        if self.t0 is None or t <= self.t0:
            lr = 1.0 * self.a / (self.tc + t ** self.beta)
        else:
            lr = 1.0 * self.a / (self.tc + self.t0 ** self.beta + t - self.t0)

        return np.clip(lr, a_min=self.lr_min, a_max=self.lr_max)

    def __str__(self):
        return 'a={} beta={} t0={} min={} max={} tc={}'.format(
            self.a, self.beta, self.t0, self.lr_min, self.lr_max, self.tc)
