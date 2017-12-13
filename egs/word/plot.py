import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from base import *


def read_log(fname):
    return wb.log_load_column(fname, to_type=float, line_head='step=')


def load_baseline():
    fname = 'data/models_ppl.txt'
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

workdirs = ['.',
            '/mnt/workspace2/wangbin/server12_work/TRF-NN-tensorflow/egs/ptb_chime4test/local',
            '/mnt/workspace/wangbin/server9_work/TRF-NN-tensorflow/egs/ptb_chime4test/local']


def search_file(name):
    for workdir in workdirs:
        s = os.path.join(workdir, name)
        if wb.exists(s):
            print('load %s' % s)
            return s
    raise TypeError('Can not find file: %s' % name)


logs = [
    'trf_nce/trf_nce10_featg4_noise2gram',
    'trf_sa/trf_sa100_featg4',
    'trf_nce/trf_nce10_e16_cnn_(1to5)x16_(3x16)x3_relu_noise2gram',
    # 'trf_sa/trf_sa100_e16_cnn_(1to5)x16_(3x16)x3_relu',
    ]
baseline_name = ['KN3', 'KN4']
colors = ['r', 'g', 'b', 'k', 'c', 'y']
baseline = wb.FRes('data/models_ppl.txt')


def smooth(a, width=1):
    b = np.array(a)
    for i in range(len(a)):
        b[i] = np.mean(a[max(0, i-width): i+1])
    return b


def subsample(a, skip=1):
    i = np.arange(0, len(a)-1, step=skip)
    return np.array(a)[i]


def plot_wer():
    x_time = False

    plt.figure()
    max_x_value = 1
    for log, color in zip(logs, colors):
        values = load_wer(search_file(log))

        if len(values['epoch']) == 0:
            raise TypeError('empty!')

        log_values = read_log(os.path.join(log, 'trf.log'))

        epochs = values['epoch']
        times = []
        for n in epochs:
            for i in range(len(log_values['epoch'])):
                if log_values['epoch'][i] >= n or i == len(log_values['epoch'])-1:
                    times.append(log_values['time'][i])
                    break

        if x_time:
            x_value = times
        else:
            x_value = epochs

        max_x_value = max(max_x_value, x_value[-1])

        plt.plot(x_value, values['wer'], color + '-', label=log)
        plt.title('wer')
        if x_time:
            plt.xlabel('time(m)')
        else:
            plt.xlabel('epoch')

    for n, name in enumerate(baseline_name):
        color = colors[n]
        x = np.linspace(0, max_x_value, 5)

        plt.plot(x, baseline.GetValue(name, 'WER') * np.ones_like(x), color + '-.s', label=name)
        plt.legend(fontsize='x-small')


def plot_ll():
    plt.figure()
    max_epoch = 1
    for log, color in zip(logs, colors):
        v = read_log(search_file(log + '/trf.log'))
        epochs = v['epoch']
        times = v['time']
        train = v['train']
        valid = v['valid']

        if len(epochs) == 0:
            print('[Warning] %s is empty!' % logs)
            continue

        max_epoch = max(max_epoch, epochs[-1])

        plt.plot(epochs, smooth(train), color+'-', label=log + '-train')
        # plt.plot(epochs, smooth(valid), color + '--', label=log + '-valid')
        plt.title('nll')
        plt.xlabel('epoch')

    for n, name in enumerate(baseline_name):
        color = colors[n]
        x = np.linspace(0, max_epoch, 2)

        if baseline.GetValue(name, 'LL-train') == 0:
            continue
        plt.plot(x, baseline.GetValue(name, 'LL-train')*np.ones_like(x), color + '-.s', label=name + '-train')
        # plt.plot(x, baseline.GetValue(name, 'LL-valid')*np.ones_like(x), color + '-.*', label=name + '-valid')
        plt.legend(fontsize='x-small')


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


def plot_logz():
    plt.figure()

    max_epoch = 0
    max_step = 0
    for log, color in zip(logs, colors):
        logname = os.path.join(log, 'logz.dbg')
        steps = []
        epochs = []
        true_logz = []
        nce_logz = []
        with open(logname) as f:
            for line in f:
                if line.find('step=') == 0:
                    a = line.split()
                    steps.append(int(a[0].split('=')[-1]))
                    epochs.append(float(a[1].split('=')[-1]))
                elif line.find('true=') == 0:
                    true_logz.append([float(s) for s in line.split()[1:]])
                elif line.find('nce=') == 0:
                    nce_logz.append([float(s) for s in line.split()[1:]])
                else:
                    raise TypeError('read line= ' + line)

        max_step = max(max_step, max(steps))
        max_epoch = max(max_epoch, max(epochs))

        true_logz = np.array(true_logz)
        nce_logz = np.array(nce_logz)[:, 0:3]

        diff = np.sum((nce_logz - true_logz) ** 2, axis=-1)
        plt.plot(epochs, diff, color, label=log)

        # diff = nce_logz - true_logz
        # ltype = ['--', ':', '-.']
        # for i in range(0, 3):
        #     plt.plot(epochs, diff[:, i], color + ltype[i], label=log + '.%d' % (i+1))

    plt.plot(np.linspace(0, max_epoch, 2), [0, 0], 'k-.')
    plt.xlabel('epoch')
    plt.ylabel('squared error')
    # plt.ylim([-0.1, 3])
    plt.xlim([0, 9])
    plt.legend()


if __name__ == '__main__':
    plot_ll()
    plot_wer()
    # plot_logz()

    plt.show()



