import tensorflow as tf

from base import *
from lm import *


def create_simulater(config, data, device='/gpu:0'):
    if config.auxiliary_type.lower() == 'lstm':
        return SimulaterLSTM(config.auxiliary_config, device)
    elif config.auxiliary_type.lower().find('gram') != -1:
        order = int(config.auxiliary_type[0:1])
        return SimulaterNgramFixed(config, data, order)
    else:
        raise TypeError('Unknown auxiliary type: {}'.format(config.auxiliary_type))


class Simulater(object):
    """
    Simulater is a auxiliary distribution used in Joint SA algorithms
    It is used to propose a sequence based on the current seqeunce,
    which helps to accelerate the MCMC sampling in SA algorithms
    """
    def __init__(self):
        # recode the time cost
        self.time_recoder = wb.clock()

    def update(self, sample_list):
        """update the simulater given a list of sequences"""
        pass

    def local_jump_propose(self, inputs, lengths, next_lengths):
        """
        Args:
            inputs: current sequences
            lengths: np.array length of current sequences
            next_lengths: np.array length of next length > lengths

        Returns:
            nexts
        """
        pass

    def local_jump_condition(self, inputs, lengths, next_lengths):
        """next_lengths < lengths"""
        pass

    def markov_move_propose(self, inputs, lengths, beg_pos, end_pos):
        pass

    def markov_move_condition(self, inputs, lengths, beg_pos, end_pos):
        pass

    def eval(self, seq_list):
        pass


class SimulaterLSTM(Simulater):
    def __init__(self, config, device, name='simulater'):
        super().__init__()
        self.model = lstmlm.LM(config, device=device, name=name)

        self.context_size = 5
        self.seqs_cache = []
        self.write_files = wb.FileBank('simulater')

    def update(self, sample_list):
        with self.time_recoder.recode('update'):
            for x in sample_list:
                self.seqs_cache.append(x[1:])  # rm the beg-tokens

            # self.model.sequence_update(session, seq_list)

            x_list, y_list = reader.produce_data_for_rnn(self.seqs_cache,
                                                         self.model.config.batch_size,
                                                         self.model.config.step_size,
                                                         include_residual_data=False)
            if len(x_list) > 0:
                for x, y in zip(x_list, y_list):
                    self.model.update(tf.get_default_session(), x, y)

                self.seqs_cache = []

    def local_jump_propose(self, inputs, lengths, next_lengths):
        if isinstance(lengths, int):
            lengths = np.array([lengths] * len(inputs))
        if isinstance(next_lengths, int):
            next_lengths = np.array([next_lengths] * len(inputs))

        assert np.all(next_lengths > lengths)

        max_next_lenght = np.max(next_lengths)
        y_list = []
        logp_list = []
        beg = 0
        for i in range(1, len(inputs)+1):
            if i == len(inputs) or lengths[i] != lengths[beg]:
                input_len = lengths[beg]
                y, y_logp = self.model.simulate(tf.get_default_session(), inputs[beg:i, 0:input_len],
                                                next_lengths[beg: i] - input_len,
                                                initial_state=True, context_size=self.context_size)
                y_list.append(np.pad(y, [[0, 0], [0, max_next_lenght-y.shape[1]]], 'edge'))
                logp_list.append(y_logp)

                beg = i

        res_y = np.concatenate(y_list, axis=0)
        res_logp = np.concatenate(logp_list, axis=0)

        return res_y, res_logp

    def local_jump_condition(self, inputs, lengths, next_lengths):
        if isinstance(lengths, int):
            lengths = np.array([lengths] * len(inputs))
        if isinstance(next_lengths, int):
            next_lengths = np.array([next_lengths] * len(inputs))

        assert np.all(next_lengths < lengths)

        return self.model.conditional(tf.get_default_session(), inputs, next_lengths, lengths,
                                      initial_state=True, context_size=self.context_size)

    def markov_move_propose(self, inputs, lengths, beg_pos, end_pos):
        sample_nums = np.minimum(lengths, end_pos)-beg_pos
        nexts, logps = self.model.simulate(tf.get_default_session(), inputs[:, 0: beg_pos], sample_nums,
                                           initial_state=True, context_size=self.context_size)
        nexts = np.concatenate([nexts, inputs[:, nexts.shape[1]:]], axis=-1)
        return nexts, logps

    def markov_move_condition(self, inputs, lengths, beg_pos, end_pos):
        return self.model.conditional(tf.get_default_session(), inputs, beg_pos, np.minimum(lengths, end_pos),
                                      initial_state=True, context_size=self.context_size)

    def eval(self, seq_list):
        """return (NLL, PPL)"""
        with self.time_recoder.recode('eval'):
            res = self.model.eval(tf.get_default_session(), seq_list, net=self.model.valid_net)
        return res


class SimulaterNgramFixed(Simulater):
    def __init__(self, config, data, order):
        super().__init__()

        self.ngram = ngram.Ngram(order, config.vocab_size)
        self.ngram.create_from_corpus(data.datas[0])

    def local_jump_propose(self, inputs, lengths, next_lengths):
        x_list = reader.extract_data_from_array(inputs, lengths)

        y_list, y_logp = self.ngram.generate(x_list, np.array(next_lengths) - np.array(lengths))

        res_y, _ = reader.produce_data_to_array(y_list)
        res_logp = np.array(y_logp)

        return res_y, res_logp

    def local_jump_condition(self, inputs, lengths, next_lengths):
        x_list = reader.extract_data_from_array(inputs, lengths)
        logp = self.ngram.condition(x_list, next_lengths)

        return np.array(logp)

    def markov_move_propose(self, inputs, lengths, beg_pos, end_pos):
        sample_nums = np.minimum(lengths, end_pos)-beg_pos
        x_list = reader.extract_data_from_array(inputs, beg_pos)
        y_list, y_logp = self.ngram.generate(x_list, sample_nums)
        nexts, _ = reader.produce_data_to_array(y_list)

        nexts = np.concatenate([nexts, inputs[:, nexts.shape[1]:]], axis=-1)
        return nexts, np.array(y_logp)

    def markov_move_condition(self, inputs, lengths, beg_pos, end_pos):
        x_list = reader.extract_data_from_array(inputs, lengths)
        logp = self.ngram.condition(x_list, beg_pos, np.minimum(lengths, end_pos))

        return np.array(logp)

    def eval(self, seq_list):
        """return (NLL, PPL)"""
        return self.ngram.eval(seq_list)

