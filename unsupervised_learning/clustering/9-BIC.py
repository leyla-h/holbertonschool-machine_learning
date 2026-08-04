#!/usr/bin/env python3
"""Finds the best number of clusters for a GMM using BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the
    Bayesian Information Criterion

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of
        clusters to check for (inclusive)
    kmax is a positive integer containing the maximum number of
        clusters to check for (inclusive)
        If kmax is None, kmax should be set to the maximum
            number of clusters possible
    iterations is a positive integer containing the maximum
        number of iterations for the EM algorithm
    tol is a non-negative float containing the tolerance for the
        EM algorithm
    verbose is a boolean that determines if the EM algorithm
        should print information to the standard output

    Returns: best_k, best_result, l, b, or None, None, None, None
        on failure
        best_k is the best value for k based on its BIC
        best_result is tuple containing pi, m, S
            pi is a numpy.ndarray of shape (k,) containing the
                cluster priors for the best number of clusters
            m is a numpy.ndarray of shape (k, d) containing the
                centroid means for the best number of clusters
            S is a numpy.ndarray of shape (k, d, d) containing
                the covariance matrices for the best number of
                clusters
        l is a numpy.ndarray of shape (kmax - kmin + 1) containing
            the log likelihood for each cluster size tested
        b is a numpy.ndarray of shape (kmax - kmin + 1) containing
            the BIC value for each cluster size tested
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    results = []
    likelihoods = []
    bics = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None

        p = (k * d * (d + 1) / 2) + (d * k) + (k - 1)
        bic = p * np.log(n) - 2 * log_likelihood

        results.append((pi, m, S))
        likelihoods.append(log_likelihood)
        bics.append(bic)

    likelihoods = np.array(likelihoods)
    bics = np.array(bics)

    best_index = np.argmin(bics)
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, likelihoods, bics
