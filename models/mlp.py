import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras import regularizers
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

def build_mlp_classifier(input_dim, n_classes, n_layers=2, dim=128,
                         lr=1e-3, l2=0., dropout=0.5, batch_norm=True):
    model = Sequential()

    model.add(Dense(dim,
                    input_dim=input_dim,
                    kernel_regularizer=regularizers.l2(l2)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout > 0.:
        model.add(Dropout(dropout))

    for _ in range(1, n_layers):
        model.add(Dense(dim,
                        kernel_regularizer=regularizers.l2(l2)))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        if dropout > 0.:
            model.add(Dropout(dropout))

    if n_classes > 2:
        model.add(Dense(n_classes,
                        kernel_regularizer=regularizers.l2(l2),
                        activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
    else:
        model.add(Dense(1,
                        kernel_regularizer=regularizers.l2(l2),
                        activation='sigmoid'))
        loss = 'binary_crossentropy'

    model.compile(loss=loss,
                  optimizer=optimizers.Adam(lr=lr),
                  metrics=['accuracy'])

    # print model.summary()

    return model


class DirMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, is_dir, n_layers=2, dim=128, lr=1e-3, l2=0.,
                 dropout=0.5, batch_norm=True, batch_size=32, epochs=1,
                 callbacks=[], verbose=0):
        self.is_dir = is_dir
        self.n_layers = n_layers
        self.dim = dim
        self.lr = lr
        self.l2 = l2
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks

        self.mlp_ = None

    def preprocess(self, X):
        def flatten(l):
            return [item for sublist in l for item in sublist]

        ret = np.asarray(X)

        if len(ret.shape) == 1:
            return self.preprocess(np.array([ret]))[0]

        ret = [flatten([[z] if not d else [(np.cos(2.0 * np.pi * z) + 1) / 2.,
                                           (np.sin(2.0 * np.pi * z) + 1) / 2.]
                        for (z, d) in zip(x, self.is_dir)])
               for x in ret]
        ret = np.asarray(ret)

        return ret

    def fit(self, X, y):
        Xprep = self.preprocess(X)

        input_dim = Xprep.shape[1]
        n_classes = len(np.unique(y))
        self.mlp_ = KerasClassifier(
            build_fn=lambda: build_mlp_classifier(
                input_dim,
                n_classes,
                n_layers=self.n_layers,
                dim=self.dim, lr=self.lr,
                l2=self.l2,
                dropout=self.dropout,
                batch_norm=self.batch_norm),
            batch_size=self.batch_size,
            verbose=self.verbose)

        return self.mlp_.fit(Xprep, y, epochs=self.epochs, callbacks=self.callbacks)

    def predict(self, X):
        Xprep = self.preprocess(X)

        return self.mlp_.predict(Xprep)
