#!/usr/bin/env python3
"""Module for Bayesian Optimization implementation."""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Represents a Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True
    ):
        """Class constructor."""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample location using Expected Improvement."""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            Y_opt = np.min(self.gp.Y)
            improvement = Y_opt - mu - self.xsi
        else:
            Y_opt = np.max(self.gp.Y)
            improvement = mu - Y_opt - self.xsi

        with np.errstate(divide='warn'):
            Z = improvement / sigma
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function."""
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Check if the next point has already been sampled
            if np.any(np.all(np.isclose(self.gp.X, X_next), axis=1)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        return self.gp.X[idx], self.gp.Y[idx]
