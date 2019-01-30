import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
import numpy as np
import copy

from models import *


"""-------------------------------------------------------------------------
   |                          General classifier                           |
   -------------------------------------------------------------------------"""


class classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def train(self, instances, labels):
        return self

    def fit(self, instances, labels):
        self.train(instances, labels)
        return self

    def predict(self, instance):
        if len(instance) == 1:
            return 0.0
        else:
            return [0.0] * instance.shape[0]

    def reduce_columns(self, index):
        pass

    def score(self, samples, labels):
        predictions = self.predict(samples)
        return accuracy_score(labels, predictions)

    def error(self, samples, labels):
        ret = 0
        predictions = np.asarray(self.predict(samples))
        predictions = predictions.reshape(len(labels), 1)
        labels = labels.reshape(len(labels), 1)
        for p, l in zip(predictions, labels):
            if p != l:
                ret += 1.0

        return ret / len(labels)

    def get_samples_by_class(self, instances, labels):
            samples_by_class = {}
            for x, t in zip(instances, labels):
                if samples_by_class.get(t, None) is None:
                    samples_by_class[t] = []
                samples_by_class[t] += [x]
            return samples_by_class

    def draw(self, samples, labels, plot=True):
        s = self.get_samples_by_class(samples, labels)

        min_v = min(x[1] for x in samples)
        max_v = max(x[1] for x in samples)
        for cl in s:
            seq = s[cl]
            plt.plot([x[1] for x in seq],
                     [(0.51 if cl >= 0.5 else 0.49) for v in seq], 'o')

        plt.xlim([min_v, max_v])

        if plot:
            plt.show()

    def draw_f(self, samples, labels, plot=True):
        if plot:
            plt.show()

        return False

    def add_bias(self, instances):
        return np.insert(instances, 0, values=1.0, axis=-1)

class sklearn_classifier(classifier):
    def __init__(self, model):
        self.model = model

    def train(self, instances, labels):
        self.model = copy.deepcopy(self.model).fit(instances, labels)
        return self

    def predict(self, instances):
        return self.model.predict(instances)

"""-------------------------------------------------------------------------
   |                         Gaussian Naive Bayes                          |
   -------------------------------------------------------------------------"""


class gaussian_naive_bayes(classifier):
    def __init__(self):
        self.gnb = GaussianNB()

    def train(self, instances, labels):
        self.gnb.fit(instances, labels)

    def predict(self, instances):
        return self.gnb.predict(instances)


"""-------------------------------------------------------------------------
   |                             Naive Bayes                               |
   -------------------------------------------------------------------------"""


class naive_bayes(classifier):
    def __init__(self, distributions=[]):
        self.distr_per_class = {}
        self.p_class = {}
        self.labels = []
        self.distributions = list(distributions)

    def reduce_columns(self, index):
        self.distributions = [self.distributions[i] for i in index]

    def set_distributions(self, distr):
        self.distributions = distr

    def get_params(self, deep=True):
        return {"distributions": self.distributions}

    def train(self, instances, labels):
        if len(instances) == 0:
            return

        # Split the samples according to their classes
        samples_by_class = self.get_samples_by_class(instances, labels)
        self.labels = list(set(labels))

        # Compute the probability of each class P(class)
        n = 0
        for k in samples_by_class:
            v = len(samples_by_class[k])
            n += v
            self.p_class[k] = v
        for k in samples_by_class:
            self.p_class[k] = float(self.p_class[k]) / float(n)

        # Compute the distribution for each class
        for k in samples_by_class:
            next_distrs = []
            for idx in range(len(self.distributions)):
                nextd = copy.deepcopy(self.distributions[idx])
                nextd.fit([x[idx] for x in samples_by_class[k]])
                next_distrs.append(nextd)
            self.distr_per_class[k] = next_distrs

    def predict(self, instances):
        def predict_instance(instance):
            ps = [np.prod([d.p(i) for (d, i) in zip(self.distr_per_class[cl],
                                                    instance)
                          ]
                         ) * self.p_class[cl] for cl in self.labels]
            ps = zip(ps, self.labels)
            return max(ps, key=lambda x: x[0])[1]

        if len(instances.shape) == 1:
            return predict_instance(instances)

        return np.asarray([predict_instance(i) for i in instances])

    def draw(self, samples, labels, plot=True):
        super(naive_bayes, self).draw(samples, labels, False)
        for cl in self.labels:
            for k in self.distr_per_class:
                distr = self.distr_per_class[k][0]
                x = np.arange(-np.pi, np.pi, 0.01)
                x = [[a] for a in x]
                t = map(lambda x: distr.p(x), x)
                plt.plot(x, t, color='r')
        plt.xlim([0, 1.0])
        plt.ylim(ymin=-0.05)
        if plot:
            plt.show()
