#!/usr/bin/env python3
""" Function that performs gradient descent with Dropout
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Y: one-hot numpy.ndarray of shape (classes, m)
    weights: dictionary of the weights and biases
    cache: dictionary of the outputs and dropout masks
    alpha: learning rate
    keep_prob: probability that a node will be kept
    L: number of layers
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        w_key = 'W' + str(i)
        b_key = 'b' + str(i)
        a_prev = cache['A' + str(i - 1)]

        dw = (1 / m) * np.matmul(dz, a_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if i > 1:
            dz = np.matmul(weights[w_key].T, dz) * (1 - np.square(a_prev))
            mask = cache['D' + str(i - 1)]
            dz = (dz * mask) / keep_prob

        weights[w_key] -= alpha * dw
        weights[b_key] -= alpha * db
