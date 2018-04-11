# using viterbi to approximate the p(x)

from base import *
from . import trfx
from .trfx import DefaultOps


class Config(trfx.Config):
    def __init__(self, data):
        super().__init__(data)

        self.init_logz = self.get_initial_logz(np.log(self.word_vocab_size))

    def __str__(self):
        s = super().__str__()
        s = s.replace('hrf_', 'hrf_app_')
        return s


class TRF(trfx.TRF):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):
        super().__init__(config, data, logdir, device, name)

    def logpxs(self, x_list, is_norm=True, for_eval=True, logzs=None):
        # get the optimized tags
        tag_list, _ = self.get_tag(x_list)
        seq_list = seq.list_create(x_list, tag_list)
        if is_norm:
            return self.logps(seq_list, for_eval)
        else:
            return self.phi(seq_list)

    def get_logpxs(self, x_list, is_norm=True, for_eval=True, batch_size=100):
        logpx = np.zeros(len(x_list))

        for i in range(0, len(x_list), batch_size):
            logpx[i: i+batch_size] = self.logpxs(x_list[i: i+batch_size], is_norm, for_eval)
        return logpx

    def rescore(self, x_list):
        return -self.get_logpxs(x_list)

    def draw(self, n):
        self.sampler.reset_dbg()

        word_list = []
        for i in range(n // self.config.chain_num):
            self.sample_seq = self.sample(*self.sample_seq)
            word_list += reader.extract_data_from_trf(*self.sample_seq)  # copy the sequence

        self.sampler.update_dbg()

        # for each word sequence, generate the tags
        tag_list, _ = self.get_tag(word_list)
        seq_list = seq.list_create(word_list, tag_list)

        with self.time_recoder.recode('write_sample'):
            f = self.write_files.get('sample')
            for s in seq_list:
                f.write(str(s))

        return seq_list

    def update(self, data_list, sample_list):
        # compute the scalars
        data_scalar = np.ones(len(data_list)) / len(data_list)
        sample_len = np.array([len(x) for x in sample_list])
        sample_facter = np.array(self.config.pi_true[self.config.min_len:]) / \
                        np.array(self.config.pi_0[self.config.min_len:])
        sample_scalar = sample_facter[sample_len - self.config.min_len] / len(sample_list)

        # update word phi
        if not self.config.fix_trf_model:
            with self.time_recoder.recode('update_word'):
                self.phi_word.update(data_list, data_scalar, sample_list, sample_scalar,
                                     learning_rate=self.cur_lr_word)

        if not self.config.fix_crf_model:
            data_x_list = [s.x[0] for s in data_list]
            with self.time_recoder.recode('update_marginal_data'):
                if self.config.semi_supervised:
                    data_tag_list, _ = self.get_tag(data_x_list)
                    data_list = seq.list_create(data_x_list, data_tag_list)

            with self.time_recoder.recode('update_tag'):
                self.phi_tag.update(data_list, data_scalar, sample_list, sample_scalar,
                                    learning_rate=self.cur_lr_tag)

            with self.time_recoder.recode('update_mix'):
                self.phi_mix.update(data_list, data_scalar, sample_list, sample_scalar,
                                    learning_rate=self.cur_lr_mix)

        # update zeta
        with self.time_recoder.recode('update_logz'):
            self.norm_const.update(sample_list, learning_rate=self.cur_lr_logz)
            # logz1 = self.get_true_logz(self.config.min_len)[0]
            # self.norm_const.set_logz1(logz1)

        # update simulater
        with self.time_recoder.recode('update_simulater'):
            self.sampler.update(seq.get_x(sample_list))

        # update dbg info
        self.sample_cur_pi.fill(0)
        for x in sample_list:
            self.sample_cur_pi[len(x)] += 1
        self.sample_acc_count += self.sample_cur_pi
        self.sample_cur_pi /= self.sample_cur_pi.sum()

        dbg_info = dict()
        acc_pi = self.sample_acc_count / np.sum(self.sample_acc_count)
        dbg_info['pi_dist'] = np.arccos(np.dot(acc_pi, self.config.pi_0) /
                                        np.linalg.norm(acc_pi) / np.linalg.norm(self.config.pi_0))

        return dbg_info
