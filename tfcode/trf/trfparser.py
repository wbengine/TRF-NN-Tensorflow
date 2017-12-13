import numpy as np
import os
import time

from . import trfbase
from .trfbase import to_str
from . import trfngram
from . import reader
from . import parser
from . import trie
from . import wblib as wb


class TRF(trfbase.FastTRF):
    def __init__(self, config, data, name='TRF', logdir='trf', simulater_device=None):
        super().__init__(config, data=data, name=name, logdir=logdir, simulater_device=simulater_device)

        self.parser_weight = None
        self.parser_bigram = trie.trie()
        self.parser_train_expec = None
        self.parser_sample_expec = None

    def create_parser_feat(self, seq_list):
        if self.parser_weight is not None:
            return

        print('exact parser features...')
        num = 0

        text_list = self.data.seqs_to_text(seq_list, skip_end_token=True, skip_beg_token=True)
        out_list, _ = parser.dependency_parser(text_list, word_to_id=self.data.word_to_id)
        for out in out_list:
            for a in out:
                sub = self.parser_bigram.setdefault([a.parent_id, a.child_id], value=num)
                if sub.data == num:
                    num += 1

        self.parser_weight = np.zeros(num)
        self.parser_train_expec = np.zeros(num)
        self.parser_sample_expec = np.zeros(num)
        self.parser_opt = wb.ArrayUpdate(self.parser_weight, {'name': 'adam'})
        print('load parser features:', num)

    def parser_find(self, seq_list):
        find_id_batch = []

        text_list = self.data.seqs_to_text(seq_list, skip_end_token=True, skip_beg_token=True)
        out_list, _ = parser.dependency_parser(text_list, word_to_id=self.data.word_to_id,
                                               logdir=self.logdir)
        for out in out_list:
            ids = []
            for a in out:
                id = self.parser_bigram.find([a.parent_id, a.child_id])
                if id is not None:
                    ids.append(id)
            find_id_batch.append(ids)
        return find_id_batch

    def save_parser(self, logdir):
        name = os.path.join(logdir, 'model.parser.feat')
        with open(name, 'wt') as f:
            num = len(self.parser_weight)
            f.write('num = {}\n'.format(num))
            write_num = trfngram.trie_write(f, self.parser_bigram, self.parser_weight)
            assert(write_num == num)

    def load_parser(self, logdir):
        name = os.path.join(logdir, 'model.parser.feat')
        if not wb.exists(name):
            return

        with open(name, 'rt') as f:
            num = int(f.__next__().split()[-1])
            self.parser_weight = np.zeros(num)
            self.parser_train_expec = np.zeros(num)
            self.parser_sample_expec = np.zeros(num)
            self.parser_opt = wb.ArrayUpdate(self.parser_weight, {'name': 'adam'})

            self.parser_bigram.clean()
            trfngram.trie_read(f, num, self.parser_bigram, self.parser_weight)

    def phi(self, inputs, lengths):
        weights = super().phi(inputs, lengths)

        seqs_list = reader.extract_data_from_trf(inputs, lengths)
        find_id_batch = self.parser_find(seqs_list)
        for i, ids in enumerate(find_id_batch):
            weights[i] += np.sum(self.parser_weight[np.array(ids, dtype='int64')])

        return weights

    def get_log_probs(self, seq_list, is_norm=True):
        logprobs = self.phi(*reader.produce_data_to_trf(seq_list))

        if is_norm:
            n = [len(x) for x in seq_list]
            logprobs += np.log(self.config.pi_0[n]) - self.logz[n]
        return logprobs

    def update(self, session, train_seqs, sample_seqs, param_lr, zeta_lr):
        super().update(session, train_seqs, sample_seqs, param_lr, zeta_lr)

        train_scalars = 1.0 / len(train_seqs) * np.ones(len(train_seqs))
        sample_n = np.array([len(x) for x in sample_seqs])
        sample_scalars = 1.0 / len(sample_seqs) * self.config.pi_true[sample_n] / self.config.pi_0[sample_n]

        self.parser_train_expec.fill(0)
        for d, idxs in zip(train_scalars, self.parser_find(train_seqs)):
            for i in idxs:
                self.parser_train_expec[i] += d

        self.parser_sample_expec.fill(0)
        for d, idxs in zip(sample_scalars, self.parser_find(sample_seqs)):
            for i in idxs:
                self.parser_sample_expec[i] += d

        grads = self.parser_sample_expec - self.parser_train_expec
        # + self.config.L2_reg * self.parser_weight

        # update
        self.parser_weight += self.parser_opt.update(grads, param_lr)

    def train(self, session, sv, nbest=None, nbest_list=None,
              print_per_epoch=0., wer_per_epoch=1., eval_list=None):

        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]
        test_list = self.data.datas[2]

        clk = wb.clock()
        print('[TRF] train_list={:,}'.format(len(train_list)),
              'valid_list={:,}'.format(len(valid_list)))

        self.load_parser(self.logdir)
        self.load_feat(self.logdir)

        self.create_parser_feat(train_list)
        self.precompute_feat(train_list)

        epoch_contain_step = len(train_list) // self.config.train_batch_size
        print_next_epoch = 0
        wer_next_epoch = 0

        time_sample = 0
        time_update = 0
        time_eval = 0
        time_true_eval = 0

        time_beginning = time.time()
        model_train_nll = []
        simul_train_nll = []
        true_train_nll = []
        kl_distance = []

        print('[TRF] [Train]...')
        for cur_epoch in range(self.config.max_epoch):
            np.random.shuffle(train_list)

            # write feat before each epoch
            self.save_feat(self.logdir)
            self.save_parser(self.logdir)

            for empirical_step in range(epoch_contain_step):
                empirical_beg = empirical_step * self.config.train_batch_size
                empirical_list = train_list[empirical_beg: empirical_beg + self.config.train_batch_size]

                global_step = cur_epoch * epoch_contain_step + empirical_step + 1
                epoch = global_step / epoch_contain_step

                clk.beg()
                sample_list = self.draw(self.config.sample_batch_size)
                time_sample += clk.end()

                param_lr = self.param_lr_computer.get_lr(global_step + 1)
                zeta_lr = self.zeta_lr_computer.get_lr(global_step + 1)
                self.update(session, empirical_list, sample_list, param_lr, zeta_lr)
                time_update += clk.end()

                model_train_nll.append(self.eval(empirical_list)[0])
                simul_train_nll.append(self.simulater.eval(session, empirical_list)[0])
                kl_distance.append(self.simulater.eval(session, sample_list)[0] -
                                   self.eval_pi0(sample_list)[0])

                time_eval += clk.end()

                # print
                if epoch >= print_next_epoch:
                    print_next_epoch += print_per_epoch

                    time_since_beg = (time.time() - time_beginning) / 60

                    model_valid_nll = self.eval(valid_list)[0]
                    model_test_nll = self.eval(test_list)[0]
                    simul_valid_nll = self.simulater.eval(session, valid_list)[0]

                    print('[TRF]',
                          'step=' + to_str(global_step),
                          'epoch=' + to_str(epoch),
                          'time=' + to_str(time_since_beg),
                          'param_lr=' + to_str(param_lr, '{:.2e}'),
                          'zeta_lr=' + to_str(zeta_lr, '{:.2e}'),
                          'lj_rate=' + to_str(1.0 * self.lj_rate),
                          'mv_rate=' + to_str(1.0 * self.mv_rate),
                          'train=' + to_str(np.mean(model_train_nll[-epoch_contain_step:])),
                          'valid=' + to_str(model_valid_nll),
                          'test=' + to_str(model_test_nll),
                          'simu_train=' + to_str(np.mean(simul_train_nll[-epoch_contain_step:])),
                          'simu_valid=' + to_str(simul_valid_nll),
                          'kl_dis=' + to_str(np.mean(kl_distance[-epoch_contain_step:])),
                          end=' ', flush=True
                          )

                    logz_sams = np.array(self.logz)
                    logz_true = np.zeros_like(logz_sams)

                    if self.config.max_len <= 5 and self.config.vocab_size < 100:
                        self.true_normalize_all()
                        true_train_nll.append(self.eval(empirical_list))
                        print('train(true)=' + to_str(np.mean(true_train_nll[-epoch_contain_step:], axis=0)),
                              'valid(true)=' + to_str(self.eval(valid_list)),
                              end=' ', flush=True
                              )
                        logz_true = np.array(self.logz)
                        self.logz = np.array(logz_sams)
                        self.zeta = np.array(logz_sams - logz_sams[self.config.min_len])
                        time_true_eval += clk.end()
                        print('time=' + to_str(time_true_eval), end=' ')

                    # calculate the WER after each epoch
                    if epoch >= wer_next_epoch and nbest is not None:
                        wer_next_epoch += wer_per_epoch
                        time_local_beg = time.time()
                        nbest.lmscore = -self.get_log_probs(nbest_list)
                        wb.WriteScore(self.logdir + '/' + self.name + '.lmscore', nbest.lmscore)
                        wer = nbest.wer()
                        time_wer = time.time() - time_local_beg
                        print('wer={:.2f} lmscale={:.2f} ({:.2f}min)'.format(wer, nbest.lmscale, time_wer / 60),
                              end=' ', flush=True)

                    print('[end]')

                    # write
                    self.write_sample.write('\n'.join([str(a) for a in sample_list]) + '\n')
                    self.write_sample.flush()
                    # self.write_train.write('\n'.join([str(a) for a in train_list[batch_beg: batch_end]]) + '\n')
                    # self.write_train.flush()
                    self.logwrite_pi(logz_sams, logz_true)
                    self.write_time.write('step=' + to_str(global_step) + ' ')
                    self.write_time.write('epoch=' + to_str(epoch) + ' ')
                    self.write_time.write('time_total=' + to_str(time_since_beg) + ' ')
                    self.write_time.write('time_sample={:.2f} '.format(time_sample) +
                                          'time_local_jump={:.2f} '.format(self.local_jump_cost) +
                                          'time_markov_move={:.2f} '.format(self.markov_move_cost) +
                                          'time_update={:.2f} '.format(time_update) +
                                          'time_eval={:.2f} '.format(time_eval) +
                                          'time_eval_true={:.2f} '.format(time_true_eval) +
                                          'feat_find={:.2f} '.format(
                                              self.feat_word.find_time if self.feat_word is not None else 0.) +
                                          'feat_find={:.2f} '.format(
                                              self.feat_class.find_time if self.feat_class is not None else 0.) +
                                          '\n'
                                          )
                    self.write_time.flush()




