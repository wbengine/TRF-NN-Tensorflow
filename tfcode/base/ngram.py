import numpy as np

from base import wblib as wb
from base import sampling as sp


class Node(object):
    def __init__(self, vocab_size):
        self.nexts = dict()
        self.probs = np.zeros(vocab_size)

    def find_node(self, contexts):
        if len(contexts) == 0:
            return self

        if contexts[0] not in self.nexts:
            return None

        return self.nexts[contexts[0]].find_node(contexts[1:])

    def insert_node(self, contexts):
        if len(contexts) == 0:
            return self

        next_node = self.nexts.setdefault(contexts[0], Node(len(self.probs)))
        return next_node.insert_node(contexts[1:])

    def normalize_probs(self, zero_prob_tokens=[]):
        self.probs /= np.sum(self.probs)
        self.probs = 0.9 * self.probs + 0.1 * np.ones_like(self.probs) / len(self.probs)
        for w in zero_prob_tokens:
            self.probs[w] = 0  # set zero probs

        self.probs = sp.FastProb(self.probs / np.sum(self.probs))
        for next_node in self.nexts.values():
            next_node.normalize_probs(zero_prob_tokens)

    def print_all(self, contexts=[]):
        print(contexts, self.probs)
        for key, next_node in self.nexts.items():
            next_node.print_all(contexts + [key])


class Ngram(object):
    def __init__(self, order, vocab_size, zero_prob_token_list=[]):
        assert order >= 1
        self.order = order
        self.vocab_size = vocab_size
        self.zero_prob_token_list = zero_prob_token_list

        self.trie = Node(vocab_size)

    def create_from_corpus(self, seq_list):
        with wb.processing('create %dgram from corpus' % self.order):
            for seq in seq_list:
                for tg_i in range(len(seq)):
                    contexts = seq[max(0, tg_i-self.order+1): max(0, tg_i)]
                    target = seq[tg_i]

                    # unigram
                    node = self.trie
                    node.probs[target] += 1

                    # 2,3,4...grams
                    contexts.reverse()
                    for w in contexts:
                        node = node.insert_node([w])
                        node.probs[target] += 1

            self.trie.normalize_probs(self.zero_prob_token_list)

    def get_prob(self, contexts):
        c = list(contexts[-(self.order-1):])
        c.reverse()
        node = self.trie
        for w in c:
            next_node = node.find_node([w])
            if next_node is None:
                break
            node = next_node
        return node.probs

    def print(self):
        self.trie.print_all()

    def get_log_probs(self, seq_list):
        return self.condition(seq_list, 1)

    def eval(self, seq_list):
        logps = self.get_log_probs(seq_list)
        nll = -np.mean(logps)
        words = np.sum([len(x)-1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)

        return nll, ppl

    def generate(self, seq_list, sample_nums):
        """
        generate samples. such as:
                input: [[1,2,3], [4,5]],  [2, 1]
                output: [[1,2,3,x,x], [4,5,x]], [logp1, logp2]
        Args:
            seq_list: a list of initial sequences
            sample_nums: list/int, the sample time for each sequences

        Returns:
            a list of new sequences, the conditional logps

        """

        if isinstance(sample_nums, int):
            sample_nums = [sample_nums] * len(seq_list)

        y_list = []
        y_logp = []
        for x, n in zip(seq_list, sample_nums):
            y = list(x)
            logp = 0
            for _ in range(n):
                p = self.get_prob(y)
                w = p.sample()
                y.append(w)
                logp += np.log(p[w])
            y_list.append(y)
            y_logp.append(logp)

        return y_list, y_logp

    def condition(self, seq_list, begs, ends=None):
        """
        compute the conditional probs. For example:
            inputs: [[1,2,3,4], [4,5]], begs=[2,1], end=[3, 2]
            outputs: np.log( p([3]|[1,2]), p([5]|[4]) )
        Args:
            seq_list: a list of sequence
            begs: a list / int
            ends: a list / int / None

        Returns:
            a list of logprobs
        """
        if isinstance(begs, int):
            begs = [begs] * len(seq_list)

        if ends is None:
            ends = [len(x) for x in seq_list]
        elif isinstance(ends, int):
            ends = [ends] * len(seq_list)

        logps = []
        for x, b, e in zip(seq_list, begs, ends):
            s = 0
            for i in range(b, e):
                p = self.get_prob(x[0:i])
                s += np.log(p[x[i]])
            logps.append(s)
        return logps



























