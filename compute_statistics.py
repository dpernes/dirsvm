import argparse
import os
import warnings

import numpy as np
from sklearn import metrics as mts
from utils import read_dataset


warnings.simplefilter("ignore")


def get_args():
    dsc = "Experiments with directional strategies."
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=dsc,
                                     formatter_class=formatter)
    parser.add_argument('--dataset', metavar="D", nargs='?',
                        default='', help='Dataset name')
    parser.add_argument('--last-partition', metavar='L', type=int,
                        default=100, help='Last partition to consider')

    return parser.parse_args()


def get_models(args):
    parts = list(os.walk(os.path.join('output',
                                      'predictions',
                                      args.dataset)))[0][1]

    models = [list(os.walk(os.path.join('output',
                                        'predictions',
                                        args.dataset,
                                        p)))[0][2]
              for p in parts]

    return sorted(list(set([m for l in models for m in l])))


def get_labels(args, fold):
    _, y = read_dataset(os.path.join('partitions', args.dataset,
                                     fold), "test.csv")
    return np.asarray(y)


def get_predictions(args, fold, model):
    f = open(os.path.join('output', 'predictions', args.dataset, fold, model))
    lines = f.readlines()
    f.close()
    return np.asarray([x.strip() for x in lines])


args = get_args()
args.predictions = os.path.join('output', 'predictions', args.dataset)
path = args.predictions
partitions = sorted(list(os.walk(os.path.join('output', 'predictions',
                                              args.dataset)))[0][1])

models = ['Directional RBF Kernel SVM.csv',
          'Cosine SVM.csv',
          'Primal Symmetric Triangle SVM.csv',
          'Symmetric Triangle Kernel SVM.csv',
          'Primal Asymmetric Triangle SVM.csv',
          'Asymmetric Triangle Kernel SVM.csv']

metrics = [('accuracy', mts.accuracy_score)]

results = {metric: {m: [] for m in models} for metric, _ in metrics}

all_res = {m: [] for m, _ in metrics}

for model in models:
    for fold in sorted(partitions):
        if int(fold) > args.last_partition:
            continue

        y = get_labels(args, fold)

        if not os.path.exists(os.path.join(path, fold, model)):
            print(('Error: Model \'%s\' doesn\'t have predictions ' +
                   'for partition %s') % (model, fold))
            print(os.path.join(path, fold, model))
            continue

        p = get_predictions(args, fold, model)

        for metric_name, metric in metrics:
            results[metric_name][model].append(metric(y, p))

    res = ['$%6.2f \pm %5.1f$' % ((100 * np.mean(results[m][model]))
                                  if len(results[m][model]) > 0 else 0.0,
                                  (100 * np.std(results[m][model]))
                                  if len(results[m][model]) > 0 else 0.0)
           for m, _ in metrics]
    for i, (m, _) in zip(res, metrics):
        all_res[m].append(i)

    print '%40s | %s' % (model, ' | '.join(res))

print
print
print

for m1 in models:
    print '%40s' % m1,
    avg_wins = []
    for m2 in models:
        sign = np.sign(np.asarray(results['accuracy'][m1]) -
                       np.asarray(results['accuracy'][m2]))
        sign = sign[sign != 0]
        if len(sign) == 0:
            sign = [0]

        print '%6.2f' % np.mean(sign == 1),

        if m2 != m1:
            avg_wins.append(np.mean(sign == 1))
    print np.mean(avg_wins),
    print
print
print
print
for m, _ in metrics:
    res = [float(x[1:].strip().split()[0]) for x in all_res[m]]
    best_ = res[np.argmax(res)]
    is_best = [np.abs(r - best_) < 1e-10 for r in res]

    res = [('$\\tb{' + s[1:].strip().split()[0] + '} '
            + ' '.join(s[1:].strip().split()[1:]))
           if b else s for s, b in zip(all_res[m], is_best)]

    print ' & '.join(res) + '\\\\'
