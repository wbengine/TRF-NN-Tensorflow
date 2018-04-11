import numpy as np
import time
import json
import os


from base import *
from . import trf, pot


class Config(trf.Config):
    def __init__(self, data):
        super().__init__(data)

        self.noise_sampler = '2gram_fixlen'

        self.fixed_length = 10
        self.fixed_offset = 5
        self.min_len = self.fixed_length
        self.max_len = self.fixed_length

        self.init_logz = [self.fixed_length, self.fixed_length]
        self.pi_true = np.ones(self.fixed_length + 1)
        self.pi_0 = np.ones(self.fixed_length + 1)

    def set_fixed(self, length, offset=5):
        self.fixed_length = length
        self.fixed_offset = offset
        self.min_len = length
        self.max_len = length

    def __str__(self):
        s = super().__str__()
        return s.replace('trf_', 'trf_fixlen_')


class DefaultOps(trf.DefaultOps):
    pass


class TRF(trf.TRF):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='trf'):

        assert config.noise_sampler.find('fixlen') != -1

        super().__init__(config, data, logdir, device, name)

        self.norm_const = pot.NormOne(config, data, config.opt_logz_method)

    def seg_sentences(self, seq_list, offset=None):
        if offset is None:
            offset = self.config.fixed_offset

        seg_seqs = []
        seg_idxs = []
        for x in seq_list:
            idx = [len(seg_seqs)]
            for i in range(0, len(x)-1, offset):
                seg = list(x[i: i + self.config.fixed_length])
                seg += [self.config.end_token] * (self.config.fixed_length - len(seg))
                seg_seqs.append(seg)
            idx.append(len(seg_seqs))
            seg_idxs.append(tuple(idx))
        return seg_seqs, seg_idxs

    # def rescore(self, seq_list):
    #     return -self.get_log_probs(seq_list, offset=1)

    def phi(self, seq_list, for_eval=True):
        n = [len(x) for x in seq_list]
        assert np.all(np.array(n) == self.config.fixed_length)
        return super().phi(seq_list, for_eval)

    def logps(self, seq_list, for_eval=True):
        logp = self.phi(seq_list, for_eval)
        return logp - self.norm_const.get_logz()

    def get_log_probs(self, seq_list, is_norm=True, offset=None):
        seg_seqs, seg_idxs = self.seg_sentences(seq_list, offset)

        seg_logps = super().get_log_probs(seg_seqs, is_norm=is_norm)

        res_logps = [np.sum(seg_logps[i: j]) for i, j in seg_idxs]
        return np.array(res_logps)

    def train(self, print_per_epoch=0.1, operation=None):

        self.data.datas[0], _ = self.seg_sentences(self.data.datas[0])

        # initialize
        self.initialize()

        if self.exist_model():
            self.restore()

        train_list = self.data.datas[0]
        self.data.write_data(train_list, os.path.join(self.logdir, 'train.seg.id'))
        valid_list = self.data.datas[1]
        test_list = self.data.datas[2]

        print('[TRF] [Train]...')
        epoch_contain_step = int(len(train_list) / self.config.batch_size)

        time_beginning = time.time()
        model_train_nll = []
        model_train_loss = []

        step = self.training_info['trained_step']
        epoch = step / epoch_contain_step
        print_next_epoch = int(epoch)
        while epoch < self.config.max_epoch:

            # update training information
            self.training_info['trained_step'] = step
            self.training_info['trained_epoch'] = epoch
            self.training_info['trained_time'] = (time.time() - time_beginning) / 60

            # shuffle the data
            if step % epoch_contain_step == 0:
                np.random.shuffle(self.data.datas[0])
                train_list, _ = self.seg_sentences(self.data.datas[0])
                self.save()

            # current data sequences
            data_seqs = train_list[
                        step % epoch_contain_step * self.config.batch_size:
                        (step % epoch_contain_step + 1) * self.config.batch_size
                        ]
            # data_seqs = self.extand_data_seqs(data_seqs)

            # update parameters
            with self.time_recoder.recode('update'):
                # update lr
                self.update_lr(step, epoch)
                # update parameters
                loss, print_infos = self.update(data_seqs)

            # compute the nll on training set
            with self.time_recoder.recode('train_eval'):
                model_train_nll.append(self.eval(data_seqs)[0])
                model_train_loss.append(loss)

            # update steps
            step += 1
            epoch = step / epoch_contain_step

            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]
                    model_test_nll = self.eval(test_list)[0]

                time_since_beg = (time.time() - time_beginning) / 60

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                info['lr_feat'] = '{:.2e}'.format(self.cur_lr_feat)
                info['lr_net'] = '{:.2e}'.format(self.cur_lr_net)
                info['lr_logz'] = '{:.2e}'.format(self.cur_lr_logz)
                # info['logz1'] = self.true_logz(self.config.min_len)[0]
                info['loss'] = np.mean(model_train_loss[-epoch_contain_step:])
                info.update(print_infos)
                info['train'] = np.mean(model_train_nll[-epoch_contain_step:])
                info['valid'] = model_valid_nll
                info['test'] = model_test_nll
                log.print_line(info)

                print('[end]')

                #####################################
                # write time
                #####################################
                # write to file
                f = self.write_files.get('time')
                f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, time_since_beg))
                f.write(
                    ' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.time_recoder.items()]) + '\n')
                f.flush()

            ###########################
            # extra operations
            ###########################
            if operation is not None:
                operation.run(step, epoch)

        self.save()
        # stop the sub-process
        self.noise_sampler.release()
