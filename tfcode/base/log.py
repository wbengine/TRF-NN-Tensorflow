import numpy as np


def to_str(v, fmt='{:.3f}'):
    if isinstance(v, (int, np.int32, np.int64)):
        return str(v)
    elif isinstance(v, (float, np.float32, np.float64)):
        if np.abs(v) < 1e3 and np.abs(v) > 1e-3:
            return fmt.format(v)
        else:
            return '{:.3e}'.format(v)
    elif isinstance(v, list) or isinstance(v, tuple):
        return '[' + ','.join([fmt.format(i) for i in v]) + ']'
    elif isinstance(v, np.ndarray):
        if v.ndim == 0:
            return fmt.format(float(v))
        else:
            return '[' + ','.join([fmt.format(i) for i in v.flatten()]) + ']'
    else:
        return str(v)


def print_line(info, end=' ', skip_none=True):
    for name, v in info.items():
        if skip_none and v is None:
            continue
        print(name + '=' + to_str(v), end=end, flush=True)


def write_array(fp, a, name='array', form='{:<10.3f}'):
    """
    write to file a 1-d array. The results is:
        name= 1, 2, 3, 4, 5, 6
    Returns:
    """
    a = np.reshape(a, [-1])
    fp.write(name + '=' + ' '.join([form.format(i) for i in a]) + '\n')
    fp.flush()


def write_seq(fp, a):
    fp.write(' '.join(str(x) for x in a) + '\n')
    fp.flush()


def write_seq_to_file(seq_list, fname):
    with open(fname, 'wt') as f:
        for a in seq_list:
            write_seq(f, a)



