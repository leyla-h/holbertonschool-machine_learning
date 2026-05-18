#!/usr/bin/env python3
"""
Module to calculate a correlation matrix from a covariance matrix.
"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix.

    Parameters:
    C (numpy.ndarray): shape (d, d) containing a covariance matrix

    Returns:
    numpy.ndarray: shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Extract the diagonal elements (variances)
    variance = np.diag(C)

    # Calculate standard deviations: std = sqrt(variance)
    std = np.sqrt(variance)

    # Convert std to a 2D matrix shape to perform outer product multiplication
    # std_product[i, j] = std[i] * std[j]
    std_product = np.outer(std, std)

    # Correlation matrix: R[i, j] = C[i, j] / (std[i] * std[j])
    R = C / std_product

    return R
