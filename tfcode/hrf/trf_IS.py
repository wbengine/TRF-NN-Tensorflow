import time

from base import *
from lm import *
from . import *
from .trf import DefaultOps
from trf.nce import noise


class Config(trf.Config):
    def __init__(self, data):
        super().__init__(data)

        self.pi_0 = self.pi_true

        self.train_batch_size = 500
        self.sample_batch_size = 500

    def __str__(self):
        s = super().__str__()
        s = s.replace('hrf_', 'hrf_IS_')
        return s


class TRF(trf.TRF):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):

        crf.CRF.__init__(self, config, data, logdir, device, name)

        # phi for words
        self.phi_word = pot.create(config.word_config, data.datas[0], config.opt_word, device)

        # logZ
        self.norm_const = norm.Norm(config, data, config.opt_logz)

        noise_config = noise.Config()
        noise_config.pack_size = config.sample_batch_size
        noise_config.min_len = config.min_len
        noise_config.max_len = config.max_len
        noise_config.beg_token = config.beg_tokens[0]
        noise_config.end_token = config.end_tokens[0]
        noise_config.vocab_size = config.word_vocab_size
        noise_config.pi_true = config.pi_true

        wod_data = data.create_data(level=0)
        self.noise_sampler = noise.NoiseSamplerNgram(noise_config, wod_data, 2, is_parallel=True)

        # learning rate
        self.cur_lr_word = 1.0
        self.cur_lr_logz = 1.0

        # debug variables
        self.sample_cur_pi = np.zeros(self.config.max_len + 1)  # current pi
        self.sample_acc_count = np.zeros(self.config.max_len + 1)  # accumulated count

    def initialize(self):
        super().initialize()

        self.noise_sampler.start()

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
        logp0 = self.get_logpxs(x_list, is_norm=True)

        approx_logz = logsumexp(logp0 - logq - np.log(n))

        log_scalar = logp0 - logq - np.log(n) - approx_logz

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
            self.norm_const.set_logz1(self.get_true_logz(self.config.min_len)[0])

        # update simulater
        # with self.time_recoder.recode('update_simulater'):
        #     self.mcmc.update(sample_list)

        # update dbg info
        self.sample_cur_pi.fill(0)
        for x in sample_list:
            self.sample_cur_pi[len(x)] += 1
        self.sample_acc_count += self.sample_cur_pi
        self.sample_cur_pi /= self.sample_cur_pi.sum()

        return None

    def train(self, print_per_epoch=0.1, operation=None):

        # initialize
        self.initialize()

        if self.exist_model():
            self.restore()
        if self.config.load_crf_model is not None:
            self.restore_crf(self.config.load_crf_model)

        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]

        print('[TRF] [Train]...')
        time_beginning = time.time()
        model_train_nll = []

        self.data.train_batch_size = self.config.train_batch_size
        self.data.is_shuffle = True
        epoch_step_num = self.data.get_epoch_step_num()
        print('[TRF] epoch_step_num={}'.format(epoch_step_num))
        print('[TRF] train_list={}'.format(len(train_list)))
        print('[TRF] valid_list={}'.format(len(valid_list)))
        last_epoch = 0
        epoch = 0
        print_next_epoch = 0
        for step, data_seqs in enumerate(self.data):

            ###########################
            # extra operations
            ###########################
            if operation is not None:
                operation.run(step, epoch)

            if int(self.data.get_cur_epoch()) > last_epoch:
                self.save()
                last_epoch = int(self.data.get_cur_epoch())

            if epoch >= self.config.max_epoch:
                print('[TRF] train stop!')
                self.save()
                # operation.perform(step, epoch)
                break

            # update epoches
            epoch = self.data.get_cur_epoch()

            # update training information
            self.training_info['trained_step'] += 1
            self.training_info['trained_epoch'] = self.data.get_cur_epoch()
            self.training_info['trained_time'] = (time.time() - time_beginning) / 60

            # draw samples
            with self.time_recoder.recode('sample'):
                sample_seqs = self.draw(self.config.sample_batch_size)

            # update paramters
            with self.time_recoder.recode('update'):
                # learining rate
                self.cur_lr_word = self.config.lr_word.get_lr(step+1, epoch)
                self.cur_lr_tag = self.config.lr_tag.get_lr(step+1, epoch)
                self.cur_lr_mix = self.config.lr_mix.get_lr(step+1, epoch)
                self.cur_lr_logz = self.config.lr_logz.get_lr(step+1, epoch)
                # update
                self.update(data_seqs, sample_seqs)

            # evaulate the nll and KL-distance
            with self.time_recoder.recode('eval_train_nll'):
                nll_train = self.eval(data_seqs)[0]
                model_train_nll.append(nll_train)

            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                time_since_beg = (time.time() - time_beginning) / 60

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                info['lr_tag'] = '{:.2e}'.format(self.cur_lr_tag)
                info['lr_mix'] = '{:.2e}'.format(self.cur_lr_mix)
                info['lr_word'] = '{:.2e}'.format(self.cur_lr_word)
                info['lr_logz'] = '{:.2e}'.format(self.cur_lr_logz)
                info['train'] = np.mean(model_train_nll[-epoch_step_num:])
                # info['train_phi'] = np.mean(model_train_nll_phi[-100:])
                info['valid'] = model_valid_nll

                ##########
                true_logz = None
                if self.config.max_len <= 5:
                    true_logz = np.array(self.get_true_logz())
                    sa_logz = np.array(self.norm_const.get_logz())
                    self.norm_const.set_logz(true_logz)
                    true_nll_train = self.eval(train_list)[0]
                    self.norm_const.set_logz(sa_logz)

                    info['true_train'] = true_nll_train

                log.print_line(info)

                print('[end]')
                # self.debug_logz()

                # write time
                f = self.write_files.get('time')
                f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, time_since_beg))
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.time_recoder.items()]) + '\n')
                f.flush()

                #  write zeta, logz, pi
                self.write_log_zeta(step, true_logz)

        self.noise_sampler.release()

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
