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


def load_wer(name, log='wer_per_epoch.log'):
    with open(os.path.join(name, log), 'rt') as f:
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
    # 'trf_t1_nce/trf_nce_noise4_data0.5_e200_blstm_200x1_pred_logzlinear_LSTMLenGen',
    # 'trf_t5_nce/trf_nce_noise4_data0.1_e200_blstm_200x1_pred_logzlinear_LSTMLenGen',
    # 'trf_t10_nce/trf_nce_noise4_data0.5_e200_blstm_200x1_pred_logzlinear_LSTMLenGen',
    # 'trf_t20_nce/trf_nce_noise4_data0.5_e200_blstm_200x1_pred_logzlinear_LSTMLenGen',
    'trf_t100_nce/trf_nce_noise4_data0.5_e200_blstm_200x1_pred_logzlinear_LSTMLenGen',
    ]
baseline_name = ['t100_KN3_002']
colors = ['r', 'g', 'b', 'k', 'c', 'y']
baseline = wb.FRes('results.txt')
baseline_fake = wb.FRes('./results.txt')


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
    for i, wer_name in enumerate(['WER-test']):
        for log, color in zip(logs, colors):
            values = load_wer(log)
            if len(values['epoch']) == 0:
                raise TypeError('empty!')

            max_epoch = max(max_epoch, values['epoch'][-1])
            plt.plot(values['epoch'], smooth(values[wer_name], width=1), color + '-', label=log)

            if wb.exists(os.path.join(log, 'sampler_wer_per_epoch.log')):
                values = load_wer(log, 'sampler_wer_per_epoch.log')
                plt.plot(values['epoch'], values[wer_name], color + '--', label=log + '[sampler]')

    for n, name in enumerate(baseline_name):
        color = colors[n]
        x = np.linspace(0, max_epoch, 5)

        plt.plot(x, baseline.GetValue(name, 'WER')*np.ones_like(x), color + ':s', label=name)
        plt.legend(fontsize='x-small')
        plt.xlabel('epoch')

    return fig


def plot_tag_wer():
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

        plt.plot(x, baseline.GetValue(name, 'WER') * np.ones_like(x), color + '-.s', label=name + '-dt')
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

def plot_pi():
    plt.figure()
    log = logs[-1]

    steps = []
    sample_pi = []
    true_pi = []
    with open(os.path.join(log, 'trf.dbg.zeta')) as f:
        for line in f:
            i = line.find('=')
            if i == -1:
                continue

            label = line[0: i]
            data = line[i+1:]
            if label.find('step') == 0:
                steps.append(int(data))
            elif label.find('all_pi') == 0:
                a = [float(i) for i in data.split()]
                sample_pi.append(a)
            elif label.find('pi_0') == 0:
                a = [float(i) for i in data.split()]
                true_pi.append(a)
            else:
                pass

    sample_pi = np.array(sample_pi)
    true_pi = np.array(true_pi)

    for i in range(sample_pi.shape[1]):
        plt.plot(steps, sample_pi[:, i] - true_pi[:, i])

    plt.xlabel('steps')

    plt.show()


if __name__ == '__main__':
    print(plot_ll())
    print(plot_wer())

    plt.show()



