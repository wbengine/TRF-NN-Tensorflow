import time
from base import *
from trf.nce import trf
from trf.nce.trf import DefaultOps

from . import sampler


class Config(trf.Config):
    def __init__(self, data):
        super().__init__(data)
        self.noise_sampler = '1gram'
        self.word_average = False
        self.pi_0 = self.pi_true

        self.sampler_config = sampler.LSTM.Config(self.vocab_size, 200, 1)
        self.sampler_config.learning_rate = 0.1
        self.load_sampler = None
        self.fix_sampler = False
        self.lr_sampler = lr.LearningRateTime(1.0)

        self.add_sampler_as_prior = False

    def __str__(self):
        s = 'trf_nce_noise{}_data{}'.format(self.noise_factor, self.data_factor)
        if self.prior_model_path is not None:
            s += '_priorlm'
        if self.add_sampler_as_prior:
            s += '_samplerlm'

        if self.feat_config is not None:
            s += '_' + str(self.feat_config)
        if self.net_config is not None:
            s += '_' + str(self.net_config)

        s += '_logz' + self.norm_config

        s += '_' + str(self.sampler_config)
        return s


class TRF(trf.TRF):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):
        super().__init__(config, data, logdir, device, name)

        if isinstance(config.sampler_config, sampler.LSTMLen.Config):
            self.sampler = sampler.LSTMLen(config, device)
        elif isinstance(config.sampler_config, sampler.LSTM.Config):
            self.sampler = sampler.LSTM(config, device)

        self.sampler.len_prob = config.pi_0
        self.sampler.update_len_prob = False

        self.sampler_nll = []
        self.sampler_kl = []

        self.cur_lr_sampler = 1.0

        if self.config.add_sampler_as_prior:
            self.priorlm = self.sampler

    def initialize(self):
        super().initialize()
        # start sampler
        self.sampler.initialize()
        if self.config.load_sampler is not None:
            print('[TRF] Load Sampler from', self.config.load_sampler)
            self.sampler.restore(self.config.load_sampler)

    def update_aux(self, seq_list, model_logps, noise_logps, data_num):

        scale_data = 1.0 / data_num * np.ones(data_num)

        n = len(seq_list) - data_num
        lengths = [len(x) for x in seq_list[data_num:]]
        logw = model_logps[data_num:] - noise_logps[data_num:]

        # logw_dict = dict()
        # for j, w in zip(lengths, logw):
        #     a = logw_dict.setdefault(j, [])
        #     a.append(w)
        #
        # logzl = np.zeros(self.config.max_len + 1)
        # for j, a in logw_dict.items():
        #     logzl[j] = logsumexp(a) - np.log(len(a))  # 1/n_l * sum_x^l e^phi(x^l) / q_l(x^l)
        #
        # log_scale = logw - logzl[lengths] - np.log(n)

        log_scale = logw - logsumexp(logw)
        scale_sample = np.exp(log_scale)

        scale = np.concatenate([scale_data, scale_sample]) * 0.5

        # all training
        self.sampler.update(seq_list[0:data_num], scale_data, self.cur_lr_sampler)
        # all p_model
        # self.sampler.update(seq_list[data_num:], scale_sample)
        # both
        # self.sampler.update(seq_list, scale)

        return scale

    def logps(self, seq_list, for_eval=True):
        phi = self.phi(seq_list, for_eval)
        lengths = np.array([len(x) for x in seq_list])

        if np.any(lengths < self.config.min_len) or np.any(lengths > self.config.max_len):
            raise TypeError('min_len={}, max_len={} lens={}'.format(min(lengths), max(lengths), lengths))

        if self.config.norm_config != 'one':
            if for_eval:
                logp_m = phi + self.priorlm.get_log_probs(seq_list) + \
                         np.log(self.config.pi_true[lengths]) - self.norm_const.get_logz(lengths)
            else:
                logp_m = phi + self.priorlm.get_log_probs(seq_list) + \
                         np.log(self.config.pi_0[lengths]) - self.norm_const.get_logz(lengths)
        else:
            logp_m = phi + self.priorlm.get_log_probs(seq_list) - self.norm_const.get_logz(lengths)

        return logp_m

    def update(self, data_list):

        # self.sampler.update(data_list, np.ones(len(data_list)) / len(data_list))
        src_data_num = len(data_list)
        assert np.min([len(x) for x in data_list]) >= self.config.min_len

        # generate noise samples
        with self.time_recoder.recode('sampling'):
            data_list = data_list + self.sampler.generate(int(self.config.data_factor * len(data_list)))
            # data_list = self.sampler.add_noise(data_list)
            sample_list = self.sampler.generate(len(data_list) * self.config.noise_factor)

            seq_list = data_list + sample_list
            assert np.min([len(x) for x in seq_list]) >= self.config.min_len
            noise_logps = self.sampler.get_log_probs(seq_list)
            data_num = len(data_list)
            seq_lens = [len(x) for x in seq_list]

        with self.time_recoder.recode('loss'):
            model_logps = self.logps(seq_list, for_eval=False)  # for training to calculate the logp
            cluster_weights = self.cmp_cluster_weight(model_logps, noise_logps, data_num, seq_lens)
            # cluster_m = self.cmp_cluster_m(model_logps, noise_logps, data_num, seq_lens)

            loss_all = self.cmp_cluster_loss(model_logps, noise_logps, data_num, seq_lens)
            loss = np.sum(loss_all)

            # cluster_weights, loss = self.get_cluster_weight_and_loss(seq_list, data_num, noise_logps)

        # update phi
        self.update_phi(seq_list, cluster_weights, cluster_m=None)

        # update zeta
        with self.time_recoder.recode('update_zeta'):
            self.norm_const.update(seq_list, cluster_weights, cluster_m=None, learning_rate=self.cur_lr_logz)
            # logz0 = self.true_logz(self.config.min_len)[0]
        # self.norm_const.set_logz0(logz0)

        # update auxiliary
        # update_aux_scale = np.ones(len(seq_list))
        with self.time_recoder.recode('update_aux'):
            if not self.config.fix_sampler:
                update_aux_scale = self.update_aux(seq_list, model_logps, noise_logps, src_data_num)
            else:
                update_aux_scale = np.ones(len(seq_list))
            sampler_ll = self.sampler.eval_nll(data_list[0: src_data_num])

        if self.config.write_dbg:
            f = self.write_files.get('noise')
            f.write('step={}\n'.format(self.training_info['trained_step']))
            f.write('[d/s] [model_logp] [noise_logp] [cluster_w] [cluster_p] [ scale ] [ seq ]\n')
            for i, s in enumerate(seq_list):
                f.write('{:>5} {:<12.5f} {:<12.5f} {:<12.5f} {:<8.5f} {:<8.5f}'.format(
                    'd' if i < len(data_list) else 's',
                    model_logps[i], noise_logps[i], cluster_weights[i], np.exp(-loss_all[i]), update_aux_scale[i]))
                f.write('[' + ' '.join(str(w) for w in s) + ']\n')
            f.flush()

        # f = self.write_files.get('logz')
        # logz = self.norm_const.get_var()
        # grad = self.norm_const.get_gradient(seq_list, cluster_weights)
        # # vars = self.norm_const.get_variance(seq_list, cluster_m)
        # log.write_array(f, logz, 'logz')
        # log.write_array(f, grad, 'grad')
        # # log.write_array(f, vars, 'vars')
        # f.write('\n')
        # f.flush()

        print_infos = OrderedDict()
        print_infos['sumw'] = np.sum(np.abs(cluster_weights))
        # print_infos['true_logz0'] = logz0
        # print_infos['nce_logz0'] = self.norm_const.get_logz(self.config.min_len)

        self.sampler_nll.append(sampler_ll)
        self.sampler_kl.append(np.mean(model_logps[src_data_num:] - noise_logps[src_data_num:]))
        print_infos['lr_sampler'] = self.cur_lr_sampler
        print_infos['aux_nll'] = np.mean(self.sampler_nll[-100:])
        print_infos['kl'] = np.mean(self.sampler_kl[-100:])
        print_infos['ESR'] = np.sum(update_aux_scale >= 1e-4) / len(update_aux_scale)
        return loss, print_infos

    def update_lr(self, step, epoch):
        super().update_lr(step, epoch)
        self.cur_lr_sampler = self.config.lr_sampler.get_lr(step+1, epoch)

    def pretrain_sampler(self, max_epoch=10,
                         print_per_epoch=0.1, operation=None,
                         write_sampler=None,
                         ):

        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]

        total_num = len(train_list)
        total_batch = total_num // self.config.batch_size

        for epoch in range(max_epoch):
            np.random.shuffle(train_list)
            nll = []

            for i in range(total_batch):
                data_list = train_list[i * self.config.batch_size: i * self.config.batch_size + self.config.batch_size]
                self.sampler.update(data_list, np.ones(self.config.batch_size) / self.config.batch_size)
                nll.append(self.sampler.eval_nll(data_list))

                if i % (int(print_per_epoch * total_batch)) == 0:
                    print('epoch={:.2f} train_nll={:.2f}'.format(epoch + i / total_batch,
                                                                 np.mean(nll[-total_batch:]))
                          )

            valid_nll = self.sampler.eval_nll(valid_list)
            print('epoch={} valid_nll={:.2f}'.format(epoch + 1, valid_nll))

            if operation is not None:
                operation.run(total_batch * (epoch + 1), epoch + 1)

            if write_sampler is not None:
                self.sampler.save(write_sampler)
