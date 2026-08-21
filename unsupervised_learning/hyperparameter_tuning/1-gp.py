#!/usr/bin/env python3
"""Module for a noiseless 1D Gaussian Process with prediction capabilities."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Class constructor."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF).
        """
        sqdist = (
            np.sum(X1 ** 2, 1).reshape(-1, 1)
            + np.sum(X2 ** 2, 1)
            - 2 * np.dot(X1, X2.T)
        )
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation (variance) of points
        in a Gaussian process.
        """
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        # Inverse of K
        K_inv = np.linalg.inv(K)

        # Mean: mu = K_s.T * K_inv * Y
        mu = np.dot(K_s.T, np.dot(K_inv, self.Y)).reshape(-1)

        # Variance/Sigma: sigma = K_ss - K_s.T * K_inv * K_s
        sigma = np.diagonal(K_ss - np.dot(K_s.T, np.dot(K_inv, K_s)))

        return mu, sigma
