#!/usr/bin/env python3
"""
Module to shuffle two matrices in the same way.
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
        X: numpy.ndarray of shape (m, nx).
        Y: numpy.ndarray of shape (m, ny).

    Returns:
        The shuffled X and Y matrices.
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)

    return X[permutation], Y[permutation]
