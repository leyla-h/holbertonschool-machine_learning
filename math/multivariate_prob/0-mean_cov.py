#!/usr/bin/env python3
"""
Module to calculate the mean and covariance of a dataset.
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Parameters:
    X (numpy.ndarray): shape (n, d) containing the data set
        n is the number of data points
        d is the number of dimensions in each data point

    Returns:
    mean, cov:
        mean: a numpy.ndarray of shape (1, d) containing the mean
        cov: a numpy.ndarray of shape (d, d) containing the covariance matrix
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate the mean with shape (1, d)
    mean = np.mean(X, axis=0, keepdims=True)

    # Shift the data by subtracting the mean
    X_centered = X - mean

    # Calculate the covariance matrix using the formula: (X_centered^T * X_centered) / (n - 1)
    cov = np.matmul(X_centered.T, X_centered) / (n - 1)

    return mean, cov
