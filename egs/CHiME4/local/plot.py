import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from base import *


def read_log(fname):
    column = ['epoch',
              'train',
              'valid'
              ]
    values = wb.log_load_column(fname, column, to_type=float, line_head='step=')

    return tuple([values[k] for k in column])


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
        labels = f.readline().split()
        while len(labels) == 0:
            labels = f.readline().split()

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

logs = [
    # 'trf_nce_new/trf_nce_e200_lstm_200x2_pred_noise2gram_onValid',
    # 'trf_nce_new/trf_nce_e200_lstm_200x2_pred_noise3gram_onValid',
    # 'trf_nce_new/trf_nce_e200_lstm_200x2_pred_noise3gram_onValidGen'
    # 'trf_nce/trf_nce_noise1_data0_e200_blstm_200x1_pred_logzlinear_LSTMGen',
    # 'trf_nce/trf_nce_noise1_data1_e200_blstm_200x1_pred_logzlinear_LSTMGen',
    'trf_nce/trf_nce_noise1_data1_e512_blstm_512x2_pred_logzlinear_LSTMGen',
    ]
baseline_name = ['KN5_00000', 'lstm_e200_h200x2']
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


def plot_wer(max_fig_id):

    max_epoch = 1
    for log, color in zip(logs, colors):
        values = load_wer(log)
        if len(values['epoch']) == 0:
            raise TypeError('empty!')

        max_epoch = max(max_epoch, values['epoch'][-1])

        plt.figure(max_fig_id)
        plt.plot(values['epoch'], values['dt'], color+'-',    label=log + '-dt')
        plt.plot(values['epoch'], values['et'], color + '--', label=log + '-et')
        plt.title('wer')
        plt.xlabel('epoch')

    for n, name in enumerate(baseline_name):
        color = colors[n]
        x = np.linspace(0, max_epoch, 5)

        plt.figure(max_fig_id)
        plt.plot(x, baseline.GetValue(name, 'dt')*np.ones_like(x), color + '-.s', label=name + '-dt')
        plt.plot(x, baseline.GetValue(name, 'et')*np.ones_like(x), color + '-.*', label=name + '-et')
        plt.legend(fontsize='x-small')
        # plt.ylim([0, 200])

    return max_fig_id + 1


def plot_ll(max_fig_id):

    max_epoch = 1
    for log, color in zip(logs, colors):
        epochs, train, valid = read_log(log + '/trf.log')

        if len(epochs) == 0:
            print('[Warning] %s is empty!' % logs)
            continue

        max_epoch = max(max_epoch, epochs[-1])

        plt.figure(max_fig_id)
        plt.plot(epochs, smooth(train), color+'-', label=log + '-train')
        plt.plot(epochs, smooth(valid), color + '--', label=log + '-valid')
        plt.title('nll')
        plt.xlabel('epoch')

    for n, name in enumerate(baseline_name):
        color = colors[n]
        x = np.linspace(0, max_epoch, 2)

        plt.figure(max_fig_id)
        if baseline.GetValue(name, 'LL-train') == 0:
            continue
        plt.plot(x, baseline.GetValue(name, 'LL-train')*np.ones_like(x), color + '-.s', label=name + '-train')
        plt.plot(x, baseline.GetValue(name, 'LL-valid')*np.ones_like(x), color + '-.*', label=name + '-valid')
        plt.legend(fontsize='x-small')

    return max_fig_id + 1


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

if __name__ == '__main__':
    fid = 1
    fid = plot_ll(fid)
    fid = plot_wer(fid)
    print('fid=', fid)

    wb.mkdir('save')
    str = ['nll', 'wer']
    for i in range(1, fid):
        exten = 'png'
        fname = 'save/figure_chime4_nce.{}.{}'.format(str[i-1], exten)
        plt.figure(i)
        plt.savefig(fname, format=exten)

    plt.show()



