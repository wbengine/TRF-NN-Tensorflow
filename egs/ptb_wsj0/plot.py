import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from model import wblib as wb


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


def load_wer(name):
    with open(os.path.join(name, 'wer_per_epoch.log'), 'rt') as f:
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
    # '/mnt/workspace2/wangbin/work/con-TRF/egs/ptb_wsj0/trf_cnn/trf_e256_cnn_relu_loademb_skipconn(1x128)(2x128)(3x128)(4x128)(5x128)(6x128)(7x128)(8x128)(9x128)(10x128)_(3x128)x3_adam/',
    # 'trf_nce/trf_nce20_e256_(1to10)x128_(3x128)x3',
    # 'trf_nce/trf_nce20_e256_(1to10)x128_(3x128)x3_updatezeta',
    # 'trf_nce/trf_nce20_lstm200x2',
    # 'trf_nce/trf_nce20_lstm200x2_updatezeta',
    # 'trf_nce/trf_nce10_rnn500x2_noise2gram',
    # 'trf_nce/trf_nce20_e256_(1to10)x128_(3x128)x3_noise2gram',
    # 'trf_nce/grf_nce20_e256_(3x128)x3_noise3gram_withq',
    # 'trf_nce/grf_nce20_e256_(3x128)x3_noise2gram_withq',
    # 'trf_nce/grf_nce20_e256_(3x128)x3_noise2gram_withqd0.5',
    # 'trf_nce/grf_nce20_e256_(1to10)x128_(3x128)x3_noise2gram_withq',
    # 'trf_nce/grf_nce20_e256_(1to10)x128_(3x128)x3_noise2gram_withq_logz10',
    # 'trf_nce/grf_nce20_e256_(1to10)x128_(3x128)x3_noise3gram_withq_logz10',
    # 'trf_cnn/trf_cnn_e250_(1to10)x128_(3x128)x3_with_lstm_withBegToken_200x2',
    # 'trf_cnn/trf_cnn_e250_(1to10)x128_(3x128)x3_with_lstm_withBegToken_200x2_zeta1',
    # 'trf_cnn/trf_cnn_e250_(1to10)x128_(3x128)x3_with_lstm_withBegToken_200x2_sgd',
    # './trf_cnn/trf_cnn_BN_e250_(1to10)x128_(3x128)x3_with_lstm_withBergToken_200x2',

    # 'trf_nce/trf_nce10_rnn200x2_noise2gram_updatezeta',
    'trf_nce/trf_nce10_rnn200x2_noise2gram',
    'trf_nce/trf_nce100_rnn200x2_noise2gram',
    'trf_nce/trf_nce10_e256_(1to10)x128_(3x128)x3_relu_noise2gram',
    'trf_nce/trf_nce100_e256_(1to10)x128_(3x128)x3_relu_noise2gram',
    ]
colors = ['r', 'g', 'b', 'k', 'c', 'y']


def smooth(a, width=10):
    b = np.array(a)
    for i in range(len(a)):
        b[i] = np.mean(a[max(0, i-width): i+1])
    return b


def subsample(a, skip=1):
    i = np.arange(0, len(a)-1, step=skip)
    return np.array(a)[i]


def plot_wer(max_fig_id):
    baseline = load_baseline()
    for a in baseline:
        print(a)

    max_epoch = 1
    for log, color in zip(logs, colors):
        values = load_wer(log)
        if len(values['epoch']) == 0:
            raise TypeError('empty!')

        max_epoch = max(max_epoch, values['epoch'][-1])

        plt.figure(max_fig_id)
        plt.plot(values['epoch'], values['wer'], color+'-',    label=log)
        plt.title('wer')
        plt.xlabel('epoch')

    for n, name in enumerate(['KN4', 'KN5', 'LSTM:h250d0epoch10.run0']):
        color = colors[n]
        x = np.linspace(0, max_epoch, 2)

        plt.figure(max_fig_id)
        plt.plot(x, baseline[name][6]*np.ones_like(x), color + '-.s', label=name)
        plt.legend(fontsize='x-small')
        # plt.ylim([0, 200])

    return max_fig_id + 1


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

        # plt.figure(max_fig_id+1)
        # plt.plot(wer, color, label=log)
        # plt.title('WER')
        # plt.legend(fontsize='small')

    for n, name in enumerate(['KN4', 'KN5', 'LSTM:h250d0epoch10.run0']):
        color = colors[n]
        x = np.linspace(0, max_epoch, 5)

        plt.figure(max_fig_id)
        plt.plot(x, baseline[name][0]*np.ones_like(x), color + '-.s', label=name + '-train')
        plt.plot(x, baseline[name][1]*np.ones_like(x), color + '-.*', label=name + '-valid')
        plt.legend(fontsize='x-small')
        plt.ylim([0, 130])

        # plt.figure(max_fig_id+1)
        # plt.plot(x, baseline[name][6]*np.ones_like(x), color + '-.', label=name + '-wer')
        # plt.legend(fontsize='small')
        # plt.ylim([6, 10])

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
        fname = 'save/figure_ptb_nce.{}.{}'.format(str[i-1], exten)
        plt.figure(i)
        plt.savefig(fname, format=exten)

    plt.show()



