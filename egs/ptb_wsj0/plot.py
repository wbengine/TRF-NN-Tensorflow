import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from base import *


def load_baseline():
    fname = 'results.txt'
    res = dict()
    with open(fname, 'rt') as f:
        f.readline()
        for line in f:
            a = line.split()
            res[a[0]] = [float(i.split('+')[0]) for i in a[1:]]
    return res


def load_wer(name):
    with open(os.path.join(name, 'wer_per_epoch.log'), 'rt') as f:
        for line in f:
            if line.split():
                labels = line.split()
                break

        values = dict()
        for line in f:
            a = line.split()
            for i, v in enumerate(a):
                try:
                    v = float(v)
                    label = labels[i]
                except:
                    label = 'epoch'
                    v = float(v[5:])

                if label not in values:
                    values[label] = []
                values[label].append(v)
        return values

# workdirs = ['.',
#             '/mnt/workspace2/wangbin/server12_work/TRF-NN-tensorflow/egs/ptb_chime4test/local',
#             '/mnt/workspace/wangbin/server9_work/TRF-NN-tensorflow/egs/ptb_chime4test/local']
workdirs = ['.']


def search_file(name):
    for workdir in workdirs:
        s = os.path.join(workdir, name)
        if wb.exists(s):
            print('load %s' % s)
            return s
    raise TypeError('Can not find file: %s' % name)


logs = [
    'trf_sa/trf_sa10_e256_cnn_(1to10)x128_(3x128)x3_relu',
    # 'trf_nce/trf_nce10_e256_cnn_(1to10)x128_(3x128)x3_relu_noise2gram',
    'trf_nce/trf_nce10_e256_cnn_(1to10)x128_(3x128)x3_relu_noise2gram_samelen',
    'trf_nce/trf_nce10_e256_cnn_(1to10)x128_(3x128)x3_relu_noise2gram_samelen_adagrad',
    ]
baseline_name = ['KN5', 'lstm_e200_h200x2']
colors = ['r', 'g', 'b', 'k', 'c', 'y']
baseline = wb.FRes('results.txt')


def smooth(a, width=1):
    b = np.array(a)
    for i in range(len(a)):
        b[i] = np.mean(a[max(0, i-width): i+1])
    return b


def subsample(a, skip=1):
    i = np.arange(0, len(a)-1, step=skip)
    return np.array(a)[i]


def plot_wer():

    fig = plt.figure()

    max_epoch = 0
    for log, color in zip(logs, colors):
        values = load_wer(search_file(log))
        if len(values['epoch']) == 0:
            raise TypeError('empty!')

        max_epoch = max(max_epoch, values['epoch'][-1])

        plt.plot(values['epoch'], values['wer'], color + '-', label=log + '-dt')
        plt.title('wer')
        plt.xlabel('epoch')

    for n, name in enumerate(baseline_name):
        color = colors[n]
        x = np.linspace(0, max_epoch, 5)

        plt.plot(x, baseline.GetValue(name, 'WER')*np.ones_like(x), color + '-.s', label=name + '-dt')
        plt.legend(fontsize='x-small')

    return fig


def plot_ll():
    fig = plt.figure()

    max_epoch = 1
    for log, color in zip(logs, colors):
        values = wb.log_load_column(search_file(log + '/trf.log'), to_type=float, line_head='step=')

        if len(values['epoch']) == 0:
            print('[Warning] %s is empty!' % logs)
            continue

        max_epoch = max(max_epoch, values['epoch'][-1])
        plt.plot(values['epoch'], smooth(values['train']), color+'-', label=log + '-train')
        plt.plot(values['epoch'], smooth(values['valid']), color + '--', label=log + '-valid')
        plt.title('nll')
        plt.xlabel('epoch')

    for n, name in enumerate(baseline_name):
        color = colors[n]
        x = np.linspace(0, max_epoch, 5)

        if baseline.GetValue(name, 'LL-train') == 0:
            continue
        plt.plot(x, baseline.GetValue(name, 'LL-train')*np.ones_like(x), color + '-.s', label=name + '-train')
        plt.plot(x, baseline.GetValue(name, 'LL-valid')*np.ones_like(x), color + '-.*', label=name + '-valid')
        plt.legend(fontsize='x-small')

    return fig


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


if __name__ == '__main__':
    print(plot_ll())
    print(plot_wer())

    # wb.mkdir('save')
    # str = ['nll', 'wer']
    # for i in range(1, fid):
    #     exten = 'png'
    #     fname = 'save/figure_chime4_nce.{}.{}'.format(str[i-1], exten)
    #     plt.figure(i)
    #     plt.savefig(fname, format=exten)

    plt.show()



