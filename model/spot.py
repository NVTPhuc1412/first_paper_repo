"""
SPOT: Streaming Peaks-Over-Threshold
Original author: Alban Siffer (GNU GPLv3)
Ported and stripped of plotting/pandas dependencies for use in anomaly detection pipeline.
"""

from math import log, floor

import numpy as np
from scipy.optimize import minimize


class SPOT:
    """
    Univariate streaming peaks-over-threshold anomaly detector.

    Fits a Generalised Pareto Distribution to the tail of an initial
    calibration batch (init_data) and uses the resulting extreme quantile
    as the anomaly threshold when run on new data.

    Usage
    -----
    s = SPOT(q=1e-5)
    s.fit(init_data, test_data)
    s.initialize(level=0.98, verbose=False)
    result = s.run(dynamic=False)
    threshold = np.mean(result['thresholds'])
    """

    def __init__(self, q=1e-4):
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def fit(self, init_data, data):
        if isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, float) and 0 < init_data < 1:
            r = int(init_data * data.size)
            self.init_data = data[:r]
            data = data[r:]
        elif isinstance(init_data, int):
            self.init_data = data[:init_data]
            data = data[init_data:]
        else:
            raise ValueError(f'Unsupported init_data type: {type(init_data)}')

        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, list):
            self.data = np.array(data)
        else:
            raise ValueError(f'Unsupported data type: {type(data)}')

    def initialize(self, level=0.98, min_extrema=False, verbose=False):
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level

        level = level - floor(level)
        n_init = self.init_data.size

        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]

        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print(f'Initial threshold : {self.init_threshold}')
            print(f'Number of peaks   : {self.Nt}')
            print('Grimshaw MLE ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')
            print(f'\tgamma = {g}, sigma = {s}, L = {l}')
            print(f'Extreme quantile (p={self.proba}): {self.extreme_quantile}')

    @staticmethod
    def _roots_finder(fun, jac, bounds, npoints, method):
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            if step == 0:
                bounds, step = (0, 1e-4), 1e-5
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        else:
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def obj(X, f, jac):
            g, j = 0, np.zeros(X.shape)
            for i, x in enumerate(X):
                fx = f(x)
                g += fx ** 2
                j[i] = 2 * fx * jac(x)
            return g, j

        opt = minimize(lambda X: obj(X, fun, jac), X0,
                       method='L-BFGS-B', jac=True,
                       bounds=[bounds] * len(X0))
        return np.unique(np.round(opt.x, decimals=5))

    @staticmethod
    def _log_likelihood(Y, gamma, sigma):
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            return -n * log(sigma) - (1 + 1 / gamma) * np.log(1 + tau * Y).sum()
        else:
            return n * (1 + log(Y.mean()))

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        def u(s): return 1 + np.log(s).mean()
        def v(s): return np.mean(1 / s)
        def w(Y, t):
            s = 1 + t * Y
            return u(s) * v(s) - 1
        def jac_w(Y, t):
            s = 1 + t * Y
            us, vs = u(s), v(s)
            return us * (1/t) * (-vs + np.mean(1/s**2)) + vs * (1/t) * (1 - vs)

        Ym, YM, Ymean = self.peaks.min(), self.peaks.max(), self.peaks.mean()
        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points
        a += epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        left  = SPOT._roots_finder(lambda t: w(self.peaks, t),
                                   lambda t: jac_w(self.peaks, t),
                                   (a, -epsilon), n_points, 'regular')
        right = SPOT._roots_finder(lambda t: w(self.peaks, t),
                                   lambda t: jac_w(self.peaks, t),
                                   (b, c), n_points, 'regular')
        zeros = np.concatenate((left, right))

        gamma_best, sigma_best = 0, Ymean
        ll_best = SPOT._log_likelihood(self.peaks, 0, Ymean)

        for z in zeros:
            if abs(z) < 1e-12:
                continue
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best, sigma_best, ll_best = gamma, sigma, ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True, dynamic=True):
        if self.n > self.init_data.size:
            print('Warning: SPOT seems to have already been run.')
            return {}

        th, alarm = [], []
        for i in range(self.data.size):
            if not dynamic:
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                if self.data[i] > self.extreme_quantile:
                    if with_alarm:
                        alarm.append(i)
                    else:
                        self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                        self.Nt += 1
                        self.n += 1
                        g, s, _ = self._grimshaw()
                        self.extreme_quantile = self._quantile(g, s)
                elif self.data[i] > self.init_threshold:
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.Nt += 1
                    self.n += 1
                    g, s, _ = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                else:
                    self.n += 1
            th.append(self.extreme_quantile)

        return {'thresholds': th, 'alarms': alarm}
