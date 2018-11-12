import csv
import os

import numpy as np


def read_csv(path, filename, delimiter=','):
    f = open(os.path.join(path, filename))
    f_csv = csv.reader(f, delimiter=delimiter)
    f_csv = list(f_csv)
    f.close()
    return f_csv


def write_csv(samples, labels, types, path):
    ret = [','.join(map(str, s) + [str(l)]) for (s, l) in zip(samples, labels)]
    ret = '\n'.join(ret)
    f = open(os.path.join(path, 'data.csv'), 'w')
    f.write(ret)
    f.close()

    f = open(os.path.join(path, 'types.data'), 'w')
    f.write('\n'.join(types))
    f.close()


def read_dataset(path, filename):
    f = read_csv(path, filename)
    samples = [list(map(float, s[: -1])) for s in f]
    labels = [s[-1] for s in f]
    return np.asarray(samples), np.asarray(labels)


def read_types(path, filename):
    f = open(os.path.join(path, filename))
    ret = [x.strip() for x in f.readlines()]
    f.close()
    return ret
