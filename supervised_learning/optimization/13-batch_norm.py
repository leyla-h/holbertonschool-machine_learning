#!/usr/bin/env python3
"""
Module to perform batch normalization.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using
    batch normalization.

    Args:
        Z: numpy.ndarray of shape (m, n) to be normalized.
        gamma: numpy.ndarray of shape (1, n) containing scales.
        beta: numpy.ndarray of shape (1, n) containing offsets.
        epsilon: Small number used to avoid division by zero.

    Returns:
        The normalized Z matrix.
    """
    # Calculate the mean and variance along the batch dimension (m)
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)

    # Normalize the batch
    Z_hat = (Z - mean) / np.sqrt(var + epsilon)

    # Scale and shift
    Z_norm = (gamma * Z_hat) + beta

    return Z_norm
