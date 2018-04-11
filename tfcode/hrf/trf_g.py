
from base import *
from lm import *
from . import *
from .trf import DefaultOps
from trf.nce import noise


class Config(trf.Config):
    def __init__(self, data):
        super().__init__(data)

        del self.__dict__['pi_0']

        average_len = np.sum(np.arange(self.max_len+1) * self.pi_true)
        self.global_logz = average_len * (np.log(self.word_vocab_size) + np.log(self.tag_vocab_size))

    def __str__(self):
        s = super().__str__()
        s = s.replace('hrf_', 'grf_')
        return s


class TRF(trf.TRF):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):
        super().__init__(config, data, logdir, device, name)

        # self.norm_const = norm.NormOne(config)
        self.norm_const = norm.Norm2(config)
        self.norm_const.logz = 0

        noise_config = noise.Config()
        noise_config.pack_size = config.sample_batch_size
        noise_config.min_len = config.min_len
        noise_config.max_len = config.max_len
        noise_config.beg_token = config.beg_tokens[0]
        noise_config.end_token = config.end_tokens[0]
        noise_config.vocab_size = config.word_vocab_size
        noise_config.pi_true = config.pi_true

        wod_data = data.create_data(level=0)
        self.noise_sampler = noise.NoiseSamplerNgram(noise_config, wod_data, 2, is_parallel=False)

    def normalize(self, logp, lengths, for_eval=True):
        logp_m = logp - self.norm_const.get_logz(lengths)
        return logp_m

    def initialize(self):
        super().initialize()

        # self.phi_tag.set_params()
        # self.phi_mix.set_params()
        # self.phi_word.set_params()

    def draw(self, n):
        assert n == self.config.sample_batch_size
        sample_list = []

        while len(sample_list) < n:
            wlist, _ = self.noise_sampler.get(None)
            seq_list = [seq.Seq(x) for x in wlist]
            sample_list += seq_list

        with self.time_recoder.recode('write_sample'):
            f = self.write_files.get('sample')
            for s in sample_list:
                f.write(str(s))

        return sample_list

    def get_sample_scaler(self, sample_list):
        n = len(sample_list)
        x_list = [s.x[0] for s in sample_list]
        logq = self.noise_sampler.noise_logps(x_list)
        logp0 = self.get_logpxs(x_list, is_norm=True) + self.norm_const.logz

        approx_logz = logsumexp(logp0 - logq - np.log(n))

        log_scalar = logp0 - logq - np.log(n) - approx_logz

        # # update logz
        # self.norm_const.logz = self.norm_const.logz * 0.9 + approx_logz * 0.1

        return np.exp(log_scalar)

    def update(self, data_list, sample_list):
        # compute the scalars
        data_scalar = np.ones(len(data_list)) / len(data_list)
        # sample_scalar = np.ones(len(sample_list)) / len(sample_list)
        sample_scalar = self.get_sample_scaler(sample_list)

        # update phi
        with self.time_recoder.recode('update_word'):
            self.phi_word.update(data_list, data_scalar, sample_list, sample_scalar,
                                 learning_rate=self.cur_lr_word)

        if not self.config.fix_crf_model:
            sample_x_list = [s.x[0] for s in sample_list]
            with self.time_recoder.recode('update_marginal'):
                sample_fp_logps_list = self.marginal_logps(sample_x_list)

            with self.time_recoder.recode('update_tag'):
                self.phi_tag.update(data_list, data_scalar, sample_list, sample_scalar,
                                    sample_fp_logps_list=sample_fp_logps_list,
                                    learning_rate=self.cur_lr_tag)

            with self.time_recoder.recode('update_mix'):
                self.phi_mix.update(data_list, data_scalar, sample_list, sample_scalar,
                                    sample_fp_logps_list=sample_fp_logps_list,
                                    learning_rate=self.cur_lr_mix)

        # update zeta
        with self.time_recoder.recode('update_logz'):
            self.norm_const.update(sample_list, sample_scalar, learning_rate=self.cur_lr_logz)

        # update simulater
        with self.time_recoder.recode('update_simulater'):
            self.mcmc.update(sample_list)

        # update dbg info
        self.sample_cur_pi.fill(0)
        for x in sample_list:
            self.sample_cur_pi[len(x)] += 1
        self.sample_acc_count += self.sample_cur_pi
        self.sample_cur_pi /= self.sample_cur_pi.sum()

        return None

    def write_log_zeta(self, step, true_logz=None):
        #  write zeta, logz, pi
        f = self.write_files.get('zeta')
        f.write('step={}\n'.format(step))
        log.write_array(f, self.sample_cur_pi[self.config.min_len:], name='cur_pi')
        log.write_array(f, self.sample_acc_count[self.config.min_len:] / self.sample_acc_count.sum(), name='all_pi')
        log.write_array(f, self.config.pi_true[self.config.min_len:], name='pi_0  ')
        log.write_array(f, self.norm_const.get_logz(), name='logz  ')
        if true_logz is not None:
            log.write_array(f, true_logz, name='truez ')
