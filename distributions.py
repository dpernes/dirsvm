#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy
import sys
from scipy.stats import norm
import collections


class distribution:
    def __init__(self):
        pass

    def fit(self, instances):
        pass

    def p(self, x):
        return 0.0


class discrete_distr(distribution):
    def __init__(self):
        self.mapping = {}

    def __str__(self):
        s = {}
        for k in self.mapping:
            s[k] = float(self.mapping[k])
        return str(s)

    def fit(self, instances):
        self.mapping = collections.Counter(instances)
        n = float(len(instances))
        self.mapping = {i: float(self.mapping[i]) / n for i in self.mapping}

    def p(self, x):
        return self.mapping.get(x, 0.0)


class gaussian_distr(distribution):
    def __init__(self):
        self.mu = 0.0
        self.sigma = 1e-6

    def fit(self, instances):
        self.mu, self.sigma = norm.fit(instances)

        if self.sigma == 0.0:
            self.sigma = 1e-6

        self.sigma2 = 2.0 * self.sigma * self.sigma
        self.p_den = self.sigma * math.sqrt(2.0 * math.pi)

    def p(self, x):
        diff_mu = x - self.mu
        return math.exp(-(diff_mu * diff_mu) / self.sigma2) / self.p_den


class von_mises(distribution):
    def __init__(self):
        self.pp = 2.0
        self.mu = 0.0
        self.kappa = 0.0

    def fit(self, instances):
        def A(p, kappa):
            return scipy.special.jv(p / 2, kappa) / \
                   scipy.special.jv(p / 2 - 1, kappa)

        def approximate_kappa(k0, R, p, iterations=2):
            k_prev = k0
            k = k0
            for i in range(iterations):
                Ap_k_prev = A(p, k_prev)
                k = k_prev - (Ap_k_prev - R) / \
                             (1.0 - Ap_k_prev * Ap_k_prev - \
                              (p - 1) * Ap_k_prev / k_prev)
                k_prev = k

            return k

        n = float(len(instances))
        if n == 0:
            n = 1.0

        inst = [x * 2.0 * np.pi - np.pi for x in instances]

        # Mu Estimation
        z = map(lambda x: complex(math.cos(x), math.sin(x)), inst)
        p = sum(z) / n

        self.mu = math.atan2(p.imag, p.real)

        # Kappa Estimation
        # Based on the estimation (4) proposed by Suvrit Sra on
        # "A short note on parameter approximation for von
        #  Mises-Fisher distributions and a fast implementation of I s (x)"
        R_cos = 1.0 / n * sum(map(math.cos, inst))
        R_sin = 1.0 / n * sum(map(math.sin, inst))
        R = math.sqrt(R_cos * R_cos + R_sin * R_sin)
        self.kappa = R * (self.pp - R * R) / (1.0 + 1e-3 - R * R)
        self.kappa = approximate_kappa(self.kappa, R, 2., 100)

        if self.kappa > 500:
            self.kappa = 300.

        if self.kappa <= 0.0:
            self.kappa = 1e-6

        self.p_den = 0.0
        try:
            self.p_den = 2.0 * np.pi * scipy.special.i0(self.kappa)
        except:
            pass

        if self.p_den == float('inf'):
            self.p_den = sys.float_info.max

    def p(self, x):
        new_x = np.pi * (2.0 * x - 1.0)
        num = 0.0

        try:
            num = math.exp(self.kappa * math.cos(new_x - self.mu))
        except:
            num = sys.float_info.max

        return num / self.p_den


def types_to_distributions(types):
    def map_type(x):
        if x == "nominal":
            return discrete_distr()
        elif x == "linear":
            return gaussian_distr()
        elif x == "directional":
            return von_mises()
        return distribution()

    return map(map_type, types)
