#!/usr/bin/env python3
"""Calculates the expectation step in the EM algorithm for a GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM

    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for
        each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the
        covariance matrices for each cluster

    Returns: g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the
            posterior probabilities for each data point in each
            cluster
        l is the total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if X.shape[1] != m.shape[1] or X.shape[1] != S.shape[1]:
        return None, None
    if pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    P = np.zeros((k, n))

    for i in range(k):
        P[i] = pi[i] * pdf(X, m[i], S[i])

    total = np.sum(P, axis=0)
    g = P / total
    l = np.sum(np.log(total))

    return g, l
