#!/usr/bin/env python3
"""
Module to normalize a matrix.
"""


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Args:
        X: numpy.ndarray of shape (d, nx) to normalize.
        m: numpy.ndarray of shape (nx,) containing the mean of all features.
        s: numpy.ndarray of shape (nx,) containing the standard deviation
           of all features.

    Returns:
        The normalized X matrix.
    """
    return (X - m) / s
