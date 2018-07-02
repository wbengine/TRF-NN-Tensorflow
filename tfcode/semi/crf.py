import numpy as np
import tensorflow as tf
import os
import time
import json

from base import *
from . import mix_net


class Config(wb.Config):
    def __init__(self, data):
        Config.value_encoding_map[lr.LearningRate] = str

        self.word_vocab_size = data.get_vocab_size()
        self.tag_vocab_size = data.get_tag_size()
        self.beg_tokens = data.get_beg_tokens()  # [word_beg_token, tag_beg_token]
        self.end_tokens = data.get_end_tokens()  # [word_end_token, tag_end_token]

        # potential features
        self.mix_config = mix_net.Config(self.word_vocab_size, data.get_char_size(), self.tag_vocab_size,
                                         self.beg_tokens[1],
                                         self.end_tokens[1])  # discrete features

        # training
        self.train_batch_size = 100

        # learning rate
        self.lr_mix = lr.LearningRateEpochDelay(1.0)
        self.opt_mix = 'sgd'
        self.max_epoch = 100

        # dbg
        self.write_dbg = False

    def __str__(self):
        return 'crf_' + str(self.mix_config)


def seq_list_package(seq_list, pad_value=0):
    lengths = [len(s) for s in seq_list]
    inputs = np.ones([len(seq_list), np.max(lengths)]) * pad_value
    labels = np.ones_like(inputs) * pad_value

    for i, s in enumerate(seq_list):
        n = len(s)
        inputs[i][0:n] = s.x[0]
        labels[i][0:n] = s.x[1]

    return inputs, labels, lengths


def seq_list_unfold(inputs, labels, lengths):
    seq_list = []
    for x, y, n in zip(inputs, labels, lengths):
        seq_list.append(seq.Seq([x[0:n], y[0:n]]))
    return seq_list


class CRF(object):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='crf'):

        self.config = config
        self.data = data
        self.logdir = logdir
        self.name = name

        self.phi_mix = mix_net.Net(self.config.mix_config, is_training=True, device=device,
                                   word_to_chars=data.vocabs[0].word_to_chars)

        # learning rate
        self.cur_lr_mix = 1.0

        # training info
        self.training_info = {'trained_step': 0,
                              'trained_epoch': 0,
                              'trained_time': 0}

        # debuger
        self.write_files = wb.FileBank(os.path.join(logdir, name + '.dbg'))
        # time recorder
        self.time_recoder = wb.clock()
        # default save name
        self.default_save_name = os.path.join(self.logdir, self.name + '.mod')

    @property
    def session(self):
        return tf.get_default_session()

    def save(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[CRF] save to', fname)
        with open(fname + '.config', 'wt') as f:
            json.dump(self.training_info, f, indent=4)
            f.write('\n')
            self.config.save(f)
        self.phi_mix.save(self.session, fname)

    def restore(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[CRF] restore from', fname)
        with open(fname + '.config', 'rt') as f:
            self.training_info = wb.json_load(f)
            print(json.dumps(self.training_info, indent=2))
        self.phi_mix.restore(self.session, fname)

    def phi(self, inputs, labels, lengths):
        return self.phi_mix.run_phi(self.session, inputs, labels, lengths)

    def logz(self, inputs, lengths):
        return self.phi_mix.run_logz(self.session, inputs, lengths)

    def logps(self, inputs, labels, lengths):
        return self.phi_mix.run_logp(self.session, inputs, labels, lengths)

    def get_log_probs(self, seq_list, is_norm=True, batch_size=100):
        logps = np.zeros(len(seq_list))

        for i in range(0, len(seq_list), batch_size):
            if is_norm:
                logps[i: i + batch_size] = self.logps(*seq_list_package(seq_list[i: i + batch_size]))
            else:
                logps[i: i + batch_size] = self.phi(*seq_list_package(seq_list[i: i + batch_size]))

        return logps

    def get_tag(self, x_list, batch_size=100):
        t_list = []
        for i in range(0, len(x_list), batch_size):
            t_list += self.phi_mix.run_opt_labels(self.session,
                                                  *reader.produce_data_to_array(x_list[i: i+batch_size]))
        return t_list

    def eval(self, seq_list):
        logps = self.get_log_probs(seq_list)
        nll = -np.mean(logps)
        words = np.sum([len(x)-1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)

        return nll, ppl

    def initialize(self):
        print('[CRF] mix_param_num = {:,}'.format(self.phi_mix.run_parameter_num(self.session)))

    def update(self, data_list):
        # compute the scalars
        inputs, labels, lengths = seq_list_package(data_list)
        self.phi_mix.run_update(self.session, inputs, labels, lengths, learning_rate=self.cur_lr_mix)
        return None

    def train(self, print_per_epoch=0.1, operation=None):

        # initialize
        self.initialize()

        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]

        print('[CRF] [Train]...')
        print('train_list=', len(train_list))
        print('valid_list=', len(valid_list))
        time_beginning = time.time()
        model_train_nll = []

        self.data.train_batch_size = self.config.train_batch_size
        self.data.is_shuffle = True
        last_epoch = 0
        print_next_epoch = 0
        for step, data_seqs in enumerate(self.data):
            epoch = self.data.get_cur_epoch()

            # update training information
            self.training_info['trained_step'] += 1
            self.training_info['trained_epoch'] = self.data.get_cur_epoch()
            self.training_info['trained_time'] = (time.time() - time_beginning) / 60

            # update paramters
            with self.time_recoder.recode('update'):
                # learining rate
                self.cur_lr_mix = self.config.lr_mix.get_lr(step+1, epoch)
                # update
                self.update(data_seqs)

            # evaulate the NLL
            with self.time_recoder.recode('eval_train_nll'):
                model_train_nll.append(self.eval(data_seqs)[0])

            if epoch >= print_next_epoch:
                print_next_epoch = epoch + print_per_epoch

                time_since_beg = (time.time() - time_beginning) / 60

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]

                info = OrderedDict()
                info['step'] = step
                info['epoch'] = epoch
                info['time'] = time_since_beg
                info['lr_mix'] = '{:.2e}'.format(self.cur_lr_mix)
                info['train'] = np.mean(model_train_nll[-100:])
                info['valid'] = model_valid_nll
                log.print_line(info)

                print('[end]')

                # write time
                f = self.write_files.get('time')
                f.write('step={} epoch={:.3f} time={:.2f} '.format(step, epoch, time_since_beg))
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.items()]) + '\n')
                f.flush()

            if epoch >= self.config.max_epoch:
                print('[CRF] train stop!')
                self.save()
                if operation is not None:
                    operation.perform(step, epoch)
                break

            if int(epoch) > last_epoch:
                self.save()
                last_epoch = int(self.data.get_cur_epoch())

            ###########################
            # extra operations
            ###########################
            if operation is not None:
                operation.run(step, epoch)

    def debug_get_logpx(self):

        for set_value in [1]:
            x_list = [sp.random_seq(3, 4,
                                    self.config.word_vocab_size,
                                    self.config.beg_tokens[0],
                                    self.config.end_tokens[0]) for _ in range(3)]

            s_list = [seq.Seq(x) for x in x_list]

            logz1 = self.phi_mix.run_logz(self.session, *reader.produce_data_to_array(x_list))

            logz2 = []
            p_sum = []
            for s in s_list:
                temp_phi = []
                temp_logp = []
                tag_iter = sp.SeqIter(len(s),
                                      self.config.tag_vocab_size,
                                      self.config.beg_tokens[1],
                                      self.config.end_tokens[1])
                for tags in tag_iter:
                    s.x[1] = tags
                    temp_phi.append(self.phi(*seq_list_package([s]))[0])
                    temp_logp.append(self.logps(*seq_list_package([s]))[0])

                print(temp_phi)
                print(logsumexp(temp_phi))
                logz2.append(logsumexp(temp_phi))
                p_sum.append(np.exp(logsumexp(temp_logp)))
            print('v={}, len={}, logz1={}, logz2={}, psum={}'.format(set_value,
                                                                     [len(x) for x in x_list],
                                                                     logz1, logz2, p_sum))


class DefaultOps(wb.Operation):
    def __init__(self, m, valid_seq_list, test_seq_list):
        super().__init__()

        self.m = m
        self.seq_list_tuple = (valid_seq_list, test_seq_list)

        self.perform_next_epoch = 1.0
        self.perform_per_epoch = 1.0
        self.write_models = wb.mkdir(os.path.join(self.m.logdir, 'crf_models'))

    def perform(self, step, epoch):
        print('[Ops] performing')

        res_prec = []
        res_time = []
        for name, seq_list in zip(['valid', 'test'], self.seq_list_tuple):
            # tagging
            time_beg = time.time()
            tag_res_list = self.m.get_tag(seq.get_x(seq_list))
            tag_time = time.time() - time_beg

            # write
            log.write_seq_to_file(tag_res_list, os.path.join(self.m.logdir, 'result_%s.tag' % name))
            gold_tag = seq.get_h(seq_list)
            log.write_seq_to_file(gold_tag, os.path.join(self.m.logdir, 'result_%s.gold.tag' % name))

            # compute wer
            P, R, F = seq.tag_error(gold_tag, tag_res_list)

            res_prec.append(P)
            res_time.append(tag_time / 60)

        print('epoch={:.2f} valid={:.2f} test={:.2f} valid_time={:.2f} test_time={:.2f}'.format(
            epoch, res_prec[0], res_prec[1], res_time[0], res_time[1]
        ))

        res = wb.FRes(os.path.join(self.m.logdir, 'results_tag_err.log'))
        res_name = 'epoch%.2f' % epoch
        res.Add(res_name, ['valid', 'test'], res_prec)
























