import numpy as np


def log_sum(logprobs):
    # logprobs = np.array(logprobs)
    # s = logprobs[0]
    # for logp in logprobs[1:]:
    #     s = np.logaddexp(logp, s)
    # return s
    max_logp = np.max(logprobs)
    return np.log(np.sum(np.exp(logprobs - max_logp))) + max_logp


def log_normalize(logprobs):
    return np.array(logprobs) - log_sum(logprobs)


def linear_normalize(probs):
    return probs / np.sum(probs)


def linear_sample(probs):
    rand_p = np.random.random()  # ranging [0,1)
    s = 0
    for i, p in enumerate(probs):
        s += p
        if rand_p < s:
            return i
    return len(probs)-1


def log_sample(probs):
    return linear_sample(np.exp(probs))


def accept(prob):
    rand_p = np.random.uniform()
    if rand_p <= prob:
        return True
    return False


def accept_logp(logp):
    return accept(np.exp(min(0, logp)))


def len_jump_distribution(min_len, max_len, width=1, self_jump=False):
    """
    generate the length-jump distribution [min_len, max_len]
    :param min_len: minimum length.
    :param max_len: maximum length.
    :param width: jump width
    :param self_jump: boolean
    :return: return a 2-D array A and A[i] denotes the jump distribution of len=i
    """
    a = np.zeros((max_len+1, max_len+1))
    for i in range(min_len, max_len+1):
        for j in range(i-width, i+width+1):
            if j >= min_len and j <= max_len:
                a[i, j] = 1
        if not self_jump:
            a[i, i] = 0
        a[i] = linear_normalize(a[i])
    return a


def random_seq(min_len, max_len, vocab_size, beg_token=None, end_token=None, pi=None):
    if pi is None:
        seq_len = np.random.randint(min_len, max_len+1)
    else:
        pi[0:min_len] = 0
        pi /= pi.sum()
        seq_len = np.random.choice(max_len+1, p=pi)
    x = [0]*seq_len
    for i in range(seq_len):
        x[i] = np.random.randint(0, vocab_size)
    if beg_token is not None:
        x[0] = beg_token
    if end_token is not None:
        x[-1] = end_token
    return x


class LenIter:
    """
    iter return integer in [min_len, max_len]
    """
    def __init__(self, min_len, max_len):
        self.min_len = min_len
        self.max_len = max_len
        self.cur_len = min_len

    def next(self):
        if self.cur_len > self.max_len:
            raise StopIteration
        self.cur_len += 1
        return self.cur_len - 1

    def __iter__(self):
        return self


class SeqIter:
    """
    enumerate a sequence A with length seq_len
    """
    def __init__(self, seq_len, vocab_size, beg_token=None, end_token=None):
        self.buf = [0] * seq_len
        self.vocab_size = vocab_size
        self.range = [0, seq_len-1]
        if beg_token is not None:
            self.buf[0] = beg_token
            self.range[0] = 1
        if end_token is not None:
            self.buf[-1] = end_token
            self.range[1] = seq_len-2

    def __next__(self):
        if self.range[1] < self.range[0] or self.buf[self.range[1]] == self.vocab_size:
            raise StopIteration
        res = list(self.buf)
        # calculate next value
        self.buf[self.range[0]] += 1
        for i in range(self.range[0], self.range[1]):
            if self.buf[i] >= self.vocab_size:
                self.buf[i] -= self.vocab_size
                self.buf[i+1] += 1
            else:
                break
        return res

    def __iter__(self):
        return self


class VecIter:
    """
    a iter enumerating all the vectors whose values at each position is in [0, max_value)
    """
    def __init__(self, max_len, max_value, init_vec=None):
        self.buf = [0] * max_len
        self.max_value = max_value
        if init_vec is not None:
            self.buf = list(init_vec)

    def next(self):
        if self.buf[-1] == self.max_value:
            raise StopIteration
        res = list(self.buf)
        # calculate next value
        self.buf[0] += 1
        for i in range(len(self.buf)-1):
            if self.buf[i] >= self.max_value:
                self.buf[i] -= self.max_value
                self.buf[i+1] += 1
            else:
                break
        return res

    def __iter__(self):
        return self


class FastProb(object):
    """a class storing the 1-d distribution and can generate sample fast"""
    def __init__(self, a):
        """a is a numpy.ndarray or a list"""
        self.size = len(a)
        self.prob = np.array(a)
        self.cumprob = np.concatenate([[0], np.cumsum(self.prob)])

    def __getitem__(self, item):
        return self.prob[item]

    def __setitem__(self, key, value):
        self.prob[key] = value
        self.cumprob = np.concatenate([[0], np.cumsum(self.prob)])

    def __str__(self):
        return str(self.prob)

    def sample(self):
        rand_p = np.random.uniform()

        left = 0
        right = self.size-1
        while left < right:
            i = (left + right) // 2
            if rand_p < self.cumprob[i]:
                right = i - 1
            elif rand_p > self.cumprob[i+1]:
                left = i + 1
            else:
                return i
        return left












