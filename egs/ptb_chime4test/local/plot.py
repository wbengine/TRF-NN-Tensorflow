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
    # 'trf_nce_new/trf_nce_e200_lstm_200x2_pred_noise2gram/',
    'trf_nce_new/trf_nce_e200_lstm_200x2_pred_noise2gram_new',
    # 'trf_nce_new/trf_nce10_e200_lstm_200x2_pred_noise2gram_nopi_pretrain',
    # 'trf_nce_new/trf_nce100_e200_lstm_200x2_pred_noise2gram_pretrain',
    # 'trf_nce_new/trf_nce10_e200_blstm_200x2_pred_noise2gram',
    # 'trf_nce_new/trf_nce10_e128_cnn_(1to10)x128_(3x128)x3_relu_noise2gram_nce2',
    # 'trf_nce_new/trf_nce10_featg2_e128_cnn_(1to10)x128_(3x128)x3_relu_noise2gram',
    # 'trf_nce_new/trf_nce10_priorlm_featg3_noise2gram',
    # 'trf_nce_new/trf_nce10_priorlm_e128_cnn_(1to10)x128_(3x128)x3_relu_noise2gram',
    # 'trf_nce_new/trf_nce10_priorlm_e128_cnn_(1to10)x128_(3x128)x3_relu_blstm_128x1_at_noise2gram'
    # 'trf_nce_new/trf_nce10_priorlm_e128_cnn_(1to10)x128_(3x128)x3_relu_noise2gram_nopi',
    'trf_sa/trf_sa10_featg3',
    ]
baseline_name = ['KN5_00000', 'lstm_e200_h200x2', 'lstm_e200_h200x2_BNCE_SGD']
colors = ['r', 'g', 'b', 'k', 'c', 'y']
baseline = wb.FRes('results.txt')


def smooth(a, width=10):
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
        values = load_wer(search_file(log))
        if len(values['epoch']) == 0:
            raise TypeError('empty!')

        max_epoch = max(max_epoch, values['epoch'][-1])

        plt.figure(max_fig_id)
        # plt.plot(values['epoch'], values['wer'], color + '-', label=log + '-dt')
        plt.plot(values['epoch'], values['dt'], color+'-',    label=log + '-dt')
        # plt.plot(values['epoch'], values['et'], color + '--', label=log + '-et')
        plt.plot(values['epoch'], values['et_real'], color + ':', label=log + '-et_real')
        plt.title('wer')
        plt.xlabel('epoch')

    for n, name in enumerate(baseline_name):
        color = colors[n]
        x = np.linspace(0, max_epoch, 5)

        plt.figure(max_fig_id)
        # plt.plot(x, baseline.GetValue(name, 'dt_real') * np.ones_like(x), color + '-.s', label=name + '-dt')
        plt.plot(x, baseline.GetValue(name, 'dt')*np.ones_like(x), color + '-.s', label=name + '-dt')
        # plt.plot(x, baseline.GetValue(name, 'et')*np.ones_like(x), color + '-.*', label=name + '-et')
        plt.plot(x, baseline.GetValue(name, 'et_real') * np.ones_like(x), color + '-.*', label=name + '-et_real')
        plt.legend(fontsize='x-small')
        # plt.xlim([0, 30])

    return max_fig_id + 1


def plot_ll(max_fig_id):

    max_epoch = 1
    for log, color in zip(logs, colors):
        epochs, train, valid = read_log(search_file(log + '/trf.log'))

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



