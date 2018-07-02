import tensorflow as tf
import time
import json
import os
from collections import OrderedDict

from base import *
from lm import *
from trf.isample import sampler
from . import net


class Config(wb.Config):
    def __init__(self, data):
        Config.value_encoding_map[lr.LearningRate] = str

        self.min_len = data.get_min_len()
        self.max_len = data.get_max_len()
        self.vocab_size = data.get_vocab_size()
        self.pi_true = data.get_pi_true()
        self.pi_0 = self.pi_true
        self.beg_token = data.get_beg_token()
        self.end_token = data.get_end_token()

        # for network features
        self.net_config = net.Config(data)

        # nce
        self.batch_size = 100
        self.sample_batch_size = 100
        # self.sampler_config = sampler.LSTMLen.Config(self.vocab_size, 200, 1)
        self.sampler_config = sampler.Ngram.Config(self.vocab_size)
        self.lr_sampler = lr.LearningRateTime(1.0)

        # learning rate
        self.lr_net = lr.LearningRateEpochDelay(1.0)
        # self.lr_logz = lr.LearningRateEpochDelay(1.0)
        self.max_epoch = 100

        # dbg
        self.write_dbg = False

    def __str__(self):
        s = 'trf_fdiv'

        if self.net_config is not None:
            s += '_' + str(self.net_config)

        s += '_' + str(self.sampler_config)
        return s


class TRF(object):
    def __init__(self, config, data, logdir, device='/gpu:0', name='trf'):
        self.config = config
        self.data = data
        self.logdir = logdir
        self.name = name

        self.net = net.Mode(self.config.net_config, is_training=True, device=device, name=name + '_net')

        # noise sampler
        self.sampler = sampler.create_sampler(config, data=data, device=device,
                                              limit_vocab=config.sampler_config.vocab_size)

        # learning rate
        self.cur_lr_net = 1.0
        # self.cur_lr_logz = 1.0
        self.cur_lr_sampler = 1.0

        # training info
        self.training_info = {'trained_step': 0,
                              'trained_epoch': 0,
                              'trained_time': 0}

        # debuger
        self.write_files = wb.FileBank(os.path.join(logdir, name + '.dbg'))
        # time recorder
        self.time_recoder = wb.clock()
        self.default_save_name = os.path.join(self.logdir, self.name + '.mod')

        self.unigram = data.get_unigram()

    @property
    def global_step(self):
        return self.net.trainop.global_step

    def save(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[TRF] save to', fname)
        with open(fname + '.config', 'wt') as f:
            json.dump(self.training_info, f, indent=4)
            f.write('\n')
            self.config.save(f)
        self.net.save(fname)

    def restore(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[TRF] restore from', fname)
        with open(fname + '.config', 'rt') as f:
            self.training_info = wb.json_load(f)
            print(json.dumps(self.training_info, indent=2))
        self.net.restore(fname)

    def exist_model(self, fname=None):
        if fname is None:
            fname = self.default_save_name

        return wb.exists(fname + '.norm')

    def logps(self, seq_list):
        return self.net.get_logps(seq_list, self.sampler.get_log_probs(seq_list))

    def get_log_probs(self, seq_list):
        logps = np.zeros(len(seq_list))
        minibatch = self.config.batch_size
        for i in range(0, len(seq_list), minibatch):
            logps[i: i + minibatch] = self.logps(seq_list[i: i + minibatch])

        return logps

    def rescore(self, seq_list):
        return -self.get_log_probs(seq_list)

    def eval(self, seq_list):
        logps = self.get_log_probs(seq_list)
        nll = -np.mean(logps)
        words = np.sum([len(x)-1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)

        return nll, ppl

    def update_lr(self, step, epoch):
        self.cur_lr_net = self.config.lr_net.get_lr(step+1, epoch)
        # self.cur_lr_logz = self.config.lr_logz.get_lr(step+1, epoch)
        self.cur_lr_sampler = self.config.lr_sampler.get_lr(step+1, epoch)

    def update(self, data_list):

        # generate noise samples
        with self.time_recoder.recode('sampling'):

            sample_list = self.sampler.generate(self.config.sample_batch_size)

            seq_list = data_list + sample_list

            logps = self.get_log_probs(seq_list)
            logqs = self.sampler.get_log_probs(seq_list)
            data_num = len(data_list)

        with self.time_recoder.recode('update'):
            loss = self.net.update(seq_list, data_num, lr=self.cur_lr_net)

        # update auxiliary
        # update_aux_scale = np.ones(len(seq_list))
        with self.time_recoder.recode('update_aux'):
            self.sampler.update(data_list, np.ones(len(data_list)) / len(data_list), lr=self.cur_lr_sampler)

        if self.config.write_dbg:
            f = self.write_files.get('noise')
            f.write('step={}\n'.format(self.training_info['trained_step']))
            f.write('[d/s] [   logp   ] [   logq   ] [ loss ] [ seq ]\n')
            for i, s in enumerate(seq_list):
                f.write('{:>5} {:<12.5f} {:<12.5f} {:<12.5f} '.format(
                    'd' if i < data_num else 's',
                    logps[i], logqs[i], loss[i]))
                f.write('[' + ' '.join(str(w) for w in s) + ']\n')
            f.flush()

        print_infos = OrderedDict()
        return np.sum(loss), print_infos

    def initialize(self):
        # print the txt information
        for d, name in zip(self.data.datas, ['train', 'valid', 'test']):
            info = wb.TxtInfo(d)
            print('[TRF]', name, ':', str(info))

        # print parameters
        print('[TRF] net_num  = {:,}'.format(self.net.get_param_num()))

        # start sampler
        self.sampler.initialize()

    def train(self, print_per_epoch=0.1, operation=None):

        # initialize
        self.initialize()

        if self.exist_model():
            self.restore()

        dataIter = reader.DataIter(self.config.batch_size, self.data)
        valid_list = self.data.datas[1]
        test_list = self.data.datas[2]

        print('[TRF] [Train]...')

        time_beginning = time.time()
        model_train_nll = []
        model_train_loss = []

        step = self.training_info['trained_step']
        epoch = self.training_info['trained_epoch']

        print_next_epoch = int(epoch)
        save_next_epoch = int(epoch) + 1
        for data_seqs in dataIter:

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
            epoch = dataIter.get_epoch()

            # update training information
            self.training_info['trained_step'] = step
            self.training_info['trained_epoch'] = epoch
            self.training_info['trained_time'] = (time.time() - time_beginning) / 60

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
                info['lr_net'] = '{:.2e}'.format(self.cur_lr_net)
                # info['lr_logz'] = '{:.2e}'.format(self.cur_lr_logz)
                info['lr_sampler'] = '{:.2e}'.format(self.cur_lr_sampler)
                info['loss'] = np.mean(model_train_loss[-100:])
                info.update(print_infos)
                info['train'] = np.mean(model_train_nll[-100:])
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
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.time_recoder.items()]) + '\n')
                f.flush()

            ###########################
            # extra operations
            ###########################
            if operation is not None:
                operation.run(step, epoch)

            if epoch >= save_next_epoch:
                save_next_epoch += 1
                self.save()
