#!/usr/bin/env python3
"""Performs K-means on a dataset"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is a positive integer containing the number of clusters

    Returns: a numpy.ndarray of shape (k, d) containing the
        initialized centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    centroids = np.random.uniform(low, high, size=(k, X.shape[1]))

    return centroids


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum
        number of iterations that should be performed

    Returns: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the
            centroid means for each cluster
        clss is a numpy.ndarray of shape (n,) containing the
            index of the cluster in C that each data point
            belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    C = initialize(X, k)
    if C is None:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    for i in range(iterations):
        C_prev = C.copy()

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for j in range(k):
            if np.sum(clss == j) == 0:
                C[j] = np.random.uniform(low, high, size=(1, d))
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        if np.array_equal(C, C_prev):
            return C, clss

    return C, clss
