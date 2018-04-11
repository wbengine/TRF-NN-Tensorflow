import time
from base import *
from . import trf

from . import sampler


class Config(trf.Config):
    def __init__(self, data):
        super().__init__(data)

        self.chain_num = 25
        self.multiple_try = 4

    def __str__(self):
        s = super().__str__()
        s = s.replace('trf_IS', 'trf_sa')
        return s


class TRF(trf.TRF):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):
        super().__init__(config, data, logdir, device, name)

        self.sample_seq = [sp.random_seq(self.config.min_len,
                           self.config.max_len,
                           self.config.vocab_size,
                           beg_token=self.config.beg_token,
                           end_token=self.config.end_token,
                           pi=self.config.pi_0)
                           for _ in range(self.config.chain_num)]

        self.mcmc_times = 0
        self.mcmc_accept = 0
        self.mcmc_rate = 0

    def perform_mcmc(self):
        def cmp_logw(seqs):
            if self.config.add_sampler_as_prior:
                seq_inputs, seq_lengths = reader.produce_data_to_array(seqs)
                return self.normalize(self.phi(seq_inputs, seq_lengths), seq_lengths, use_pi_true=False)
            else:
                return self.get_log_probs(seqs, use_pi_true=False) - self.sampler.get_log_probs(seqs)

        new_seqs = self.sampler.generate(self.config.chain_num * self.config.multiple_try)
        logw = cmp_logw(new_seqs)
        logw = np.reshape(logw, [self.config.chain_num, self.config.multiple_try])
        logw_sum = logsumexp(logw, axis=-1, keepdims=True)
        logw_norm = logw - logw_sum

        next_idxs = [sp.log_sample(p) for p in logw_norm]
        next_seqs = [new_seqs[i * self.config.multiple_try + idx] for i, idx in enumerate(next_idxs)]
        next_logw = logw[np.arange(self.config.chain_num), next_idxs]
        curr_logw = cmp_logw(self.sample_seq)

        logw2 = np.array(logw)
        logw2[np.arange(self.config.chain_num), next_idxs] = curr_logw

        log_acc = logsumexp(logw, axis=-1) - logsumexp(logw2, axis=-1)

        for i, loga in enumerate(log_acc):
            if sp.accept_logp(loga):
                self.sample_seq[i] = next_seqs[i]
                self.mcmc_accept += 1
            self.mcmc_times += 1

        f = self.write_files.get('mcmc')
        out_line = 'rate=[{}/{}]'.format(self.mcmc_accept, self.mcmc_times) + \
                   ' acc=' + log.to_str(np.exp(log_acc)) + \
                   ' cur_logw=' + log.to_str(curr_logw) + \
                   ' next_logw=' + log.to_str(next_logw)
        f.write(out_line + '\n')

    def draw(self, n):

        self.mcmc_times = 0
        self.mcmc_accept = 0

        seqs = []
        while len(seqs) < n:
            self.perform_mcmc()
            for a in self.sample_seq:
                seqs.append(list(a))

        self.mcmc_rate = 0.9 * self.mcmc_rate + 0.1 * self.mcmc_accept / self.mcmc_times

        with self.time_recoder.recode('write_sample'):
            f = self.write_files.get('sample')
            for x in seqs:
                log.write_seq(f, x)
            f.flush()

        return seqs

    def update(self, data_list, sample_list):
        info = OrderedDict()
        # compute the scalars
        data_scalar = np.ones(len(data_list)) / len(data_list)
        sample_len = np.array([len(x) for x in sample_list])
        sample_facter = np.array(self.config.pi_true[self.config.min_len:]) / \
                        np.array(self.config.pi_0[self.config.min_len:])
        sample_scalar = sample_facter[sample_len - self.config.min_len] / len(sample_list)

        # update feat-phi
        with self.time_recoder.recode('update_feat'):
            self.phi_feat.update(data_list, data_scalar,
                                 sample_list, sample_scalar,
                                 learning_rate=self.cur_lr_feat)

        # update net-phi
        with self.time_recoder.recode('update_net'):
            self.phi_net.update(data_list, data_scalar,
                                sample_list, sample_scalar,
                                learning_rate=self.cur_lr_net)

        # update zeta
        with self.time_recoder.recode('update_zeta'):
            self.norm_const.update(sample_list, learning_rate=self.cur_lr_logz)
            self.norm_const.set_logz1(self.true_logz(self.config.min_len)[0])

        # update simulater
        with self.time_recoder.recode('update_sampler'):
            if not self.config.fix_sampler:
                self.sampler.update(data_list, np.ones(len(data_list)) / 100, lr=self.cur_lr_sampler, batch_size=100)

        # update dbg info
        self.sample_cur_pi.fill(0)
        for x in sample_list:
            self.sample_cur_pi[len(x)] += 1
        self.sample_acc_pi += self.sample_cur_pi
        self.sample_cur_pi /= self.sample_cur_pi.sum()

        acc_pi = self.sample_acc_pi / np.sum(self.sample_acc_pi)
        info['acc_rate'] = self.mcmc_rate
        info['pi_dist'] = np.arccos(np.dot(acc_pi, self.config.pi_0) /
                                    np.linalg.norm(acc_pi) / np.linalg.norm(self.config.pi_0))

        #  write zeta, logz, pi
        f = self.write_files.get('zeta')
        f.write('step={}\n'.format(self.training_info['trained_step']))
        log.write_array(f, self.config.pi_0[self.config.min_len:], name='pi_0  ')
        log.write_array(f, acc_pi[self.config.min_len:], name='acc_pi')
        log.write_array(f, self.sample_cur_pi[self.config.min_len:], name='cur_pi')
        log.write_array(f, self.norm_const.zeta, name='zeta  ')
        log.write_array(f, self.norm_const.get_logz(), name='logz  ')
        f.flush()

        return info

    def eval(self, seq_list):
        logps = self.get_log_probs(seq_list, minibatch=self.config.chain_num * self.config.multiple_try)
        nll = -np.mean(logps)
        words = np.sum([len(x) - 1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)

        return nll, ppl

    # def train(self, print_per_epoch=0.1, operation=None):
    #
    #     # initialize
    #     self.initialize()
    #
    #     if self.exist_model():
    #         self.restore()
    #
    #     train_list = self.data.datas[0]
    #     valid_list = self.data.datas[1]
    #     test_list = self.data.datas[2]
    #
    #     print('[TRF] [Train]...')
    #     epoch_contain_step = len(train_list) // self.config.train_batch_size
    #
    #     time_beginning = time.time()
    #     model_train_nll = []
    #     kl_distance = []
    #
    #     step = self.training_info['trained_step']
    #     epoch = step / epoch_contain_step
    #     print_next_epoch = int(epoch)
    #
    #     while epoch < self.config.max_epoch:
    #
    #         # update training information
    #         self.training_info['trained_step'] = step
    #         self.training_info['trained_epoch'] = epoch
    #         self.training_info['trained_time'] = (time.time() - time_beginning) / 60
    #
    #         if step % epoch_contain_step == 0:
    #             np.random.shuffle(train_list)
    #             self.save()
    #
    #         # get empirical list
    #         data_beg = step % epoch_contain_step * self.config.train_batch_size
    #         data_list = train_list[data_beg: data_beg + self.config.train_batch_size]
    #
    #         # draw samples
    #         with self.time_recoder.recode('sample'):
    #             sample_list = self.draw(self.config.sample_batch_size)
    #
    #         # update paramters
    #         with self.time_recoder.recode('update'):
    #             # learining rate
    #             self.cur_lr_feat = self.config.lr_feat.get_lr(step+1, epoch)
    #             self.cur_lr_net = self.config.lr_net.get_lr(step+1, epoch)
    #             self.cur_lr_logz = self.config.lr_logz.get_lr(step+1, epoch)
    #             self.cur_lr_sampler = self.config.lr_sampler.get_lr(step+1, epoch)
    #             # update
    #             self.update(data_list, sample_list)
    #
    #         ##########################
    #         # update step
    #         ##########################
    #         step += 1
    #         epoch = step / epoch_contain_step
    #
    #         # evaulate the nll and KL-distance
    #         with self.time_recoder.recode('eval_train_nll'):
    #             model_train_nll.append(self.eval(data_list)[0])
    #         with self.time_recoder.recode('eval_kl_dis'):
    #             kl_distance.append(self.simular.eval(sample_list)[0] - self.eval(sample_list, for_eval=False)[0])
    #
    #         # print
    #         if epoch >= print_next_epoch:
    #             print_next_epoch = epoch + print_per_epoch
    #
    #             time_since_beg = (time.time() - time_beginning) / 60
    #
    #             with self.time_recoder.recode('eval'):
    #                 model_valid_nll = self.eval(valid_list)[0]
    #                 # model_test_nll = self.eval(test_list)[0]
    #                 # simul_valid_nll = self.simulater.eval(session, valid_list)[0]
    #
    #             info = OrderedDict()
    #             info['step'] = step
    #             info['epoch'] = epoch
    #             info['time'] = time_since_beg
    #             info['lr_feat'] = '{:.2e}'.format(self.cur_lr_feat)
    #             info['lr_net'] = '{:.2e}'.format(self.cur_lr_net)
    #             info['lr_logz'] = '{:.2e}'.format(self.cur_lr_logz)
    #             info['lj_rate'] = self.simular.lj_rate
    #             info['mv_rate'] = self.simular.mv_rate
    #             info['train'] = np.mean(model_train_nll[-epoch_contain_step:])
    #             info['valid'] = model_valid_nll
    #             # info['test'] = model_test_nll
    #             info['kl_dis'] = np.mean(kl_distance[-epoch_contain_step:])
    #             log.print_line(info)
    #
    #             print('[end]')
    #
    #             # write time
    #             f = self.write_files.get('time')
    #             f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, time_since_beg))
    #             f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.time_recoder.items()]) + '\n')
    #             f.flush()
    #
    #             #  write zeta, logz, pi
    #             f = self.write_files.get('zeta')
    #             f.write('step={}\n'.format(step))
    #             log.write_array(f, self.sample_cur_pi[self.config.min_len:], name='cur_pi')
    #             log.write_array(f, self.sample_acc_count[self.config.min_len:]/self.sample_acc_count.sum(), name='all_pi')
    #             log.write_array(f, self.config.pi_0[self.config.min_len:], name='pi_0  ')
    #             log.write_array(f, self.norm_const.get_logz(), name='logz  ')
    #
    #         ###########################
    #         # extra operations
    #         ###########################
    #         if operation is not None:
    #             operation.run(step, epoch)
    #
