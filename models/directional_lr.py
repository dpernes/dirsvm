from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import numpy as np


class DirLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, is_directional, C=1.0):
        self.is_directional = is_directional
        self.C = C
        self.estimator = None

    def fit(self, X, y):
        Xtr = self.transform_input(X)
        self.estimator = LogisticRegression(
            C=self.C,
            random_state=42).fit(Xtr, y)

        return self

    def predict(self, X):
        return self.estimator.predict(self.transform_input(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(self.transform_input(X))

    def decision_function(self, X):
        return self.estimator.decision_function(self.transform_input(X))

    def transform_input(self, X):
        def flatten(l):
            return [item for sublist in l for item in sublist]

        ret = np.asarray(X)

        if len(ret.shape) == 1:
            return self.transform_input(np.array([ret]))[0]

        ret = [flatten([[y] if not d else [(np.cos(2.0 * np.pi * y) + 1) / 2.,
                                           (np.sin(2.0 * np.pi * y) + 1) / 2.]
                        for (y, d) in zip(x, self.is_directional)])
               for x in ret]
        ret = np.asarray(ret)

        return ret
