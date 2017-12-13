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
