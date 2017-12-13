import numpy as np
import matplotlib.pyplot as plt
import sys

from base import *


def read_log(fname):
    column = ['epoch',
              'train',
              'valid',
              'wer',
              'kl_dis'
              ]
    values = wb.log_load_column(fname, column, to_type=float, line_head='step=')

    # concatenate the epochs
    epoch_base = 0
    epoch = values[0]
    epoch_new = np.zeros_like(epoch)
    for i in range(len(epoch)):
        # if i > 0 and epoch[i] < epoch[i-1]:
        #     epoch_base = epoch[i-1]
        epoch_new[i] = epoch[i] + epoch_base

    values[0] = epoch_new.tolist()
    return tuple(values)


def load_baseline():
    fname = 'data/models_ppl.txt'
    res = dict()
    with open(fname, 'rt') as f:
        f.readline()
        for line in f:
            a = line.split()
            res[a[0]] = [float(i.split('+')[0]) for i in a[1:]]
    return res


logs = [
    # 'trf_g2_adam',
    # 'trf_g4_adam',
    # 'trf_g4_ngram_var',
    # 'trf_cnn_discrete/trf_g4_ngram_sgd',
    # 'trf_cnn_discrete/trf_g4_ngram_adam',
    # 'trf_nce/trf_nce_rnn128x1',
    'trf_nce/global_trf_nce10_e32_(1to5)x32_(3x32)x3',
    ]
colors = ['r', 'g', 'b', 'k', 'c', 'y']


def smooth(a, width=1):
    b = np.array(a)
    for i in range(len(a)):
        b[i] = np.mean(a[max(0, i-width): i+1])
    return b


def subsample(a, skip=1):
    i = np.arange(0, len(a)-1, step=skip)
    return np.array(a)[i]


def plot_ll(max_fig_id):
    baseline = load_baseline()
    print(baseline)

    max_epoch = 1
    for log, color in zip(logs, colors):
        epochs, train, valid, wer, kl = read_log(log + '/trf.log')

        if len(epochs) == 0:
            print('[Warning] %s is empty!' % logs)
            continue

        max_epoch = max(max_epoch, epochs[-1])

        plt.figure(max_fig_id)
        plt.plot(epochs, smooth(train), color+'-', label=log + '-train')
        plt.plot(epochs, smooth(valid), color + '--', label=log + '-valid')
        plt.title('nll')
        plt.xlabel('epoch')

        # plt.figure(max_fig_id+1, figsize=(10, 10))
        # plt.plot(epochs, smooth(kl, 100), color, label=log + '(sample)')
        # plt.title('KL distance')
        # plt.legend(fontsize='small')

        plt.figure(max_fig_id+1)
        plt.plot(wer, color, label=log)
        plt.title('WER')
        plt.legend(fontsize='small')

    for n, name in enumerate(['KN4', 'KN5']):
        color = colors[n]
        x = np.linspace(0, max_epoch, 5)

        plt.figure(max_fig_id)
        plt.plot(x, baseline[name][0]*np.ones_like(x), color + '-.s', label=name + '-train')
        plt.plot(x, baseline[name][1]*np.ones_like(x), color + '-.*', label=name + '-valid')
        plt.legend(fontsize='x-small')
        # plt.ylim([0, 200])

        plt.figure(max_fig_id+1)
        plt.plot(x, baseline[name][6]*np.ones_like(x), color + '-.', label=name + '-wer')
        plt.legend(fontsize='small')
        # plt.ylim([6, 10])

    return max_fig_id + 2


def plot_time():
    plt.figure()
    for log, color in zip(logs, colors):
        res = wb.log_load_column(log.replace('.log', '.time'), ['train_epoch', 'time_sample', 'time_local_jump', 'time_markov_move'], float)
        plt.plot(res[0], res[1], color+'-', label=log + '[total]')
        plt.plot(res[0], res[2], color+'--', label=log + '[local_jump]')
        plt.plot(res[0], res[3], color+':', label=log+'[markov_move]')

        plt.legend()
        plt.title('time (minute)')
        plt.xlabel('epoch')

    plt.show()


def plot_pi(max_fig_id):
    values = dict()
    with open(logs[-1].replace('.log', '.debug'), 'rt') as f:
        for line in f:
            gap = line.find('=')
            if gap == -1:
                continue
            line = line.replace('=', ' ')
            line = line.replace('inf', '0')
            a = line.split()
            name = a[0]
            data = [float(i) for i in a[1:]]

            values.setdefault(name, list())
            values[name].append(data)

    for key in values:
        values[key] = np.array(values[key])

    plt.figure(max_fig_id, figsize=(10, 10))
    plt.title(logs[-1])
    plt.subplot(1, 2, 1)
    a = (values['all_pi'] - values['pi_0']) / values['pi_0']
    plt.title('(sample_pi-true_pi)/true_pi')
    plt.plot(a)
    plt.ylim([-5, 5])

    plt.subplot(1, 2, 2)
    plt.plot(values['all_pi'][-1], label='all_pi')
    plt.plot(values['pi_0'][-1], label='pi_0')
    plt.title('sample_pi and true_pi at the last iteration')
    plt.legend()

    return max_fig_id + 1


def read_lstm_log(fname):
    epochs = []
    ppl_valid = []
    ppl_test = []
    total_time = []

    cur_epoch = 0
    cur_time = 0
    with open(fname, 'rt') as f:
        for line in f:
            if line.find('epoch=') == 0:
                a = line.split()
                cur_epoch = float(a[0].split('=')[1])
                cur_time = float(a[-1].split('=')[1][0:-1])
            elif line.find('ppl-valid=') == 0:
                epochs.append(cur_epoch)
                total_time.append(cur_time)
                a = line.split()
                ppl_valid.append(float(a[0].split('=')[1]))
                ppl_test.append(float(a[1].split('=')[1]))
    return epochs, ppl_valid, ppl_test, total_time


def plot_lstm():
    lstm_logs = ['lstmlm/lstm_v331005_e256_h2048x1_AdaptiveSoftmax']

    for log, color in zip(lstm_logs, colors):
        epochs, ppl_valid, ppl_test, total_time = read_lstm_log(log + '/lstm.log')

        if len(epochs) == 0:
            print('[Warning] %s is empty!' % logs)
            continue

        plt.plot(epochs, ppl_valid, color + '-', label=log + '-valid')
        plt.plot(epochs, ppl_test, color + '--', label=log + '-test')
        plt.title('ppl')
        plt.xlabel('epoch')
        plt.ylim([35, 60])
        plt.legend()

if __name__ == '__main__':
    # fid = 1
    # fid = plot_ll(fid)
    # print('fid=', fid)
    #
    # # wb.mkdir('save')
    # # str = ['nll', 'kl', 'wer', 'pi']
    # # for i in range(1, fid):
    # #     exten = 'pdf'
    # #     fname = 'save/figure_word_g5.{}.{}'.format(str[i-1], exten)
    # #     plt.figure(i)
    # #     plt.savefig(fname, format=exten, dpi=10)
    #
    # plt.show()

    plot_lstm()
    plt.show()



