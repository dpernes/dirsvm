import argparse
import os

import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier

from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping

from models.directional_lr import DirLogisticRegression
from models.generative import naive_bayes
from models.svm import (AsyTriKernelSVM, AsyTriSvm2step, DirectionalKernelSVM,
                        SymTriSvm)
from models.mlp import build_mlp_classifier, DirMLP
import distributions

from utils import read_dataset, read_types


def get_args():
    dsc = "Experiments with directional strategies."
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=dsc,
                                     formatter_class=formatter)
    parser.add_argument('--dataset', metavar="D", nargs='?',
                        default="", help='Path to the dataset')
    parser.add_argument('--first-partition', metavar="F", type=int,
                        default=0, help='First partition')
    parser.add_argument('--last-partition', metavar="L", type=int,
                        default=30, help='Last partition')
    parser.add_argument('--output', metavar="O", nargs='?',
                        default=os.path.join('output', 'predictions'),
                        help='Path to the output folder')

    return parser.parse_args()


def get_models(args, types, n_classes):
    is_dir = [x == "directional" for x in types]

    # von Mises Naive Bayes
    vm_types = ["linear" if t in [] else t for t in types]
    vm_distr = distributions.types_to_distributions(vm_types)
    vmnb = naive_bayes(distributions=vm_distr)

    # Directional Logistic Regression
    dlr = OneVsOneClassifier(
        GridSearchCV(
            DirLogisticRegression(is_directional=is_dir),
            param_grid={'C': np.logspace(-3, 3, 5)},
            cv=3),
        n_jobs=-1)

    # Cosine SVM
    kcos_svm = OneVsOneClassifier(
        GridSearchCV(
            DirectionalKernelSVM(is_dir, kernel='cosine'),
            param_grid={'C': np.logspace(-3, 3, 5),
                        'max_iter': np.logspace(4, 6, 5, dtype=int)},
            cv=3),
        n_jobs=-1)

    # Directional RBF Kernel SVM
    drbf_svm = OneVsOneClassifier(
        GridSearchCV(
            DirectionalKernelSVM(is_dir, kernel='drbf'),
            param_grid={'C': np.logspace(-3, 3, 5),
                        'kernel_param': np.logspace(-3, 1, 5),
                        'max_iter': np.logspace(4, 6, 5, dtype=int)},
            cv=3),
        n_jobs=-1)

    # (Primal) Symmetric Triangle SVM
    stri_svm = OneVsOneClassifier(
        GridSearchCV(
            SymTriSvm(is_directional=is_dir, optimizer='Adam',
                      num_epochs=500,),
            param_grid={'C': np.logspace(-3, 3, 5),
                        'lr': np.logspace(-5, -1, 10)},
            cv=3),
        n_jobs=-1)

    # Symmetric Triangle Kernel SVM
    ktri_svm = OneVsOneClassifier(
        GridSearchCV(
            DirectionalKernelSVM(is_dir, kernel='triangular'),
            param_grid={'C': np.logspace(-3, 3, 5),
                        'kernel_param': np.logspace(-3, 1, 5),
                        'max_iter': np.logspace(4, 6, 5, dtype=int)},
            cv=3),
        n_jobs=-1)

    # (Primal) Asymmetric Triangle SVM
    atri_svm = OneVsOneClassifier(
        GridSearchCV(
            AsyTriSvm2step(is_directional=is_dir, optimizer='Adam',
                           num_epochs1=400, num_epochs2=500),
            param_grid={'C': np.logspace(-3, 3, 5),
                        'lr': np.logspace(-5, -1, 10)},
            cv=3),
        n_jobs=-1)

    # Asymmetric Triangle Kernel SVM
    katri_svm = OneVsOneClassifier(
        GridSearchCV(
            AsyTriKernelSVM(is_directional=is_dir,
                            num_epochs=100,
                            optimizer='Adam'),
            param_grid={'C': np.logspace(-3, 3, 5),
                        'lr': np.logspace(-4, -2, 10),
                        'max_iter': np.logspace(4, 6, 5, dtype=int)},
            cv=3)
    )

    # MLP
    rmlp = GridSearchCV(
        KerasClassifier(build_fn=build_mlp_classifier,
                        batch_size=128,
                        verbose=1),
        param_grid={'input_dim': [len(types)],
                    'n_classes': [n_classes],
                    'n_layers': [4, 5],
                    'dim': [256],
                    'lr': [1e-4, 1e-3],
                    'l2': [0., 1e-4],
                    'dropout': [0., 0.2, 0.5],
                    'epochs': [200]},
        cv=3,
        n_jobs=1)

    # Directional MLP
    dmlp = GridSearchCV(
        DirMLP(is_dir,
               batch_size=128,
               callbacks=[EarlyStopping(monitor='loss',
                                        min_delta=1e-4,
                                        patience=5)]),
        param_grid={'n_layers': [4, 5],
                    'dim': [256],
                    'lr': [1e-4, 1e-3],
                    'l2': [0., 1e-4],
                    'dropout': [0., 0.2, 0.5],
                    'epochs': [200]},
        cv=3,
        n_jobs=-1)

    ret = [('dLR', dlr),
           ('vMNB', vmnb),
           ('Cosine SVM', kcos_svm),
           ('Directional RBF Kernel SVM', drbf_svm),
           ('Primal Symmetric Triangle SVM', stri_svm),
           ('Symmetric Triangle Kernel SVM', ktri_svm),
           ('Primal Asymmetric Triangle SVM', atri_svm),
           ('Asymmetric Triangle Kernel SVM', katri_svm),
           ('rMLP', rmlp),
           ('dMLP', dmlp)]

    return ret


def get_dataset(args, fold):
    tr_X, tr_y = read_dataset(os.path.join(args.dataset, '%03d' % fold),
                              "train.csv")
    ts_X, ts_y = read_dataset(os.path.join(args.dataset, '%03d' % fold),
                              "test.csv")

    return np.asarray(tr_X), np.asarray(tr_y), np.asarray(ts_X), \
        np.asarray(ts_y)


args = get_args()

types = read_types(args.dataset, "types.data")
_, tr_y, _, _ = get_dataset(args, 0)
n_classes = len(np.unique(tr_y))
models = get_models(args, types, n_classes)

tr_accuracy_values = {n: [] for n, _ in models}
ts_accuracy_values = {n: [] for n, _ in models}
f1_values = {n: [] for n, _ in models}

for foldid in range(args.first_partition, args.last_partition):
    tr_X, tr_y, ts_X, ts_y = get_dataset(args, foldid)
    print 'Fold %3d' % foldid

    output_path = os.path.join(args.output, '%03d' % foldid)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for name, model in models:
        if name == 'MLP':
            model.fit(tr_X, tr_y, callbacks=[EarlyStopping(monitor='loss',
                                                           min_delta=1e-4,
                                                           patience=5)])
        else:
            model.fit(tr_X, tr_y)
        tr_preds = model.predict(tr_X)
        ts_preds = model.predict(ts_X)

        tr_accuracy_values[name].append(metrics.accuracy_score(tr_y, tr_preds))
        ts_accuracy_values[name].append(metrics.accuracy_score(ts_y, ts_preds))
        f1_values[name].append(metrics.f1_score(ts_y,
                                                ts_preds,
                                                average='macro'))

        print '%40s %9.4f %9.4f %9.4f' % (name,
                                          100*np.mean(
                                              tr_accuracy_values[name]),
                                          100*np.mean(
                                              ts_accuracy_values[name]),
                                          100*np.mean(f1_values[name]))

        f = open(os.path.join(output_path, name + '.csv'), 'w')
        f.write('\n'.join(map(str, ts_preds)))
        f.close()

    print
