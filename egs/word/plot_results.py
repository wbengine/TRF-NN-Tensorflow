import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os

from model import wblib as wb


def read_log(fname):
    column = ['epoch',
              'train',
              'valid',
              ]
    values = wb.log_load_column(fname, column, to_type=float, line_head='step=')

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

global_colors = ['r', 'g', 'b', 'k', 'c', 'y']


def smooth(a, width=1):
    b = np.array(a)
    for i in range(len(a)):
        b[i] = np.mean(a[max(0, i-width): i+1])
    return b


def subsample(a, skip=1):
    i = np.arange(0, len(a)-1, step=skip)
    return np.array(a)[i]


def get_name(log):
    a = os.path.split(log)[-1].split('_')
    gram_name = ['unigram', 'bigram', 'trigram']
    name = '$\\nu$=' + a[1][3:] + ', $p_n$=' + gram_name[int(a[3][5:6])-1]
    if 'updatezeta' not in a:
        name += ',' + 'fix $\zeta$'
    return name


def plot_ll(logs, save_file, colors=global_colors):

    plt.figure()

    for log, color in zip(logs, colors):
        epochs, train, valid = read_log(log + '/nll.dbg')
        plt.plot(subsample(epochs, 10), subsample(valid, 10), color, label=get_name(log))

        # epochs, train, valid = read_log(log + '/trf.log')
        # plt.plot(subsample(epochs, 10), subsample(valid, 10), color+':', label=get_name(log))

        # plt.plot(epochs, smooth(valid, 100), color + '--', label=log + '-valid')
        plt.ylabel('NLL')
        plt.xlabel('epoch')
        plt.xlim([0, 9])
        plt.legend()

    plt.savefig(save_file, format='eps')


def plot_logz(logs, save_file, colors=global_colors):
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
        nce_logz = np.array(nce_logz)

        # diff = (nce_logz - true_logz)  # / true_logz
        # ltype = [':*', ':x', ':o']
        # for i in range(true_logz.shape[1]):
        #     plt.plot(subsample(epochs, 10), subsample(diff[:, i], 10), color + ltype[i])

        diff = np.sum((nce_logz - true_logz) ** 2, axis=-1)
        plt.plot(subsample(epochs, 5), subsample(diff, 5), color, label=get_name(log))

    plt.plot(np.linspace(0, max_epoch, 2), [0, 0], 'k-.')
    plt.xlabel('epoch')
    plt.ylabel('squared error')
    # plt.ylim([-0.1, 3])
    plt.xlim([0, 9])
    plt.legend()

    plt.savefig(save_file, format='eps')


if __name__ == '__main__':

    mpl.rc('font', family='Times New Roman', size=28)
    # mpl.rc('xtick', labelsize=25)
    # mpl.rc('ytick', labelsize=25)
    mpl.rc('legend', fontsize=20)
    mpl.rc('lines', linewidth=2, markersize=10)
    mpl.rc('figure.subplot', left=0.18, bottom=0.18)

    logs = [
        'trf_nce/trf_nce1_rnn16x1_noise1gram_updatezeta',
        'trf_nce/trf_nce10_rnn16x1_noise1gram_updatezeta',
        'trf_nce/trf_nce100_rnn16x1_noise1gram_updatezeta',
    ]
    plot_ll(logs, 'D:\\wangbin\\doc\\TRF2017-2\\fig\\exp1_nll_1.eps')
    plot_logz(logs, 'D:\\wangbin\\doc\\TRF2017-2\\fig\\exp1_zeta_1.eps')

    # logs = [
    #     'trf_nce/trf_nce10_rnn16x1_noise1gram_updatezeta',
    #     'trf_nce/trf_nce10_rnn16x1_noise1gram',
    # ]
    # plot_ll(logs, 'D:\\wangbin\\doc\\TRF2017-2\\fig\\exp1_nll_2.eps')
    # plot_logz(logs, 'D:\\wangbin\\doc\\TRF2017-2\\fig\\exp1_zeta_2.eps')

    logs = [
        'trf_nce/trf_nce1_rnn16x1_noise1gram_updatezeta',
        'trf_nce/trf_nce1_rnn16x1_noise2gram_updatezeta',
        'trf_nce/trf_nce10_rnn16x1_noise1gram_updatezeta',
        'trf_nce/trf_nce10_rnn16x1_noise2gram_updatezeta',
    ]
    plot_ll(logs, 'D:\\wangbin\\doc\\TRF2017-2\\fig\\exp1_nll_3.eps', colors=['r', 'r:', 'g', 'g:'])
    plot_logz(logs, 'D:\\wangbin\\doc\\TRF2017-2\\fig\\exp1_zeta_3.eps', colors=['r', 'r:', 'g', 'g:'])

    # plt.show()



