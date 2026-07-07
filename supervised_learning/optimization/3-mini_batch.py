#!/usr/bin/env python3
"""
Module to create mini-batches for training.
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training a neural network.

    Args:
        X: numpy.ndarray of shape (m, nx).
        Y: numpy.ndarray of shape (m, ny).
        batch_size: The number of data points in a batch.

    Returns:
        List of mini-batches containing tuples (X_batch, Y_batch).
    """
    X, Y = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []

    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
