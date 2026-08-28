#!/usr/bin/env python3
"""Bayesian Optimization module"""

import numpy as np
GaussianProcess = __import__('2-gaussian_process').GaussianProcess


class BayesianOptimization:
    """Represents a Bayesian optimization"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, random_sample=0):
        self.f = f
        self.gp = GaussianProcess(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.random_sample = random_sample

    def acquisition(self):
        """Calculates the acquisition function"""
        from scipy.stats import norm
        mu, sigma = self.gp.predict(self.X_s)
        mu_sample_opt = np.min(self.gp.Y)

        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - self.xsi
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function"""
        for _ in range(iterations):
            X_next, EI = self.acquisition()
            if np.any((self.gp.X == X_next).all(axis=1)):
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        idx = np.argmin(self.gp.Y)
        return self.gp.X[idx], self.gp.Y[idx].reshape(-1)
