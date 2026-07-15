#!/usr/bin/env python3
""" Function that performs forward propagation with Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    X: numpy.ndarray of shape (nx, m) containing input data
    weights: dictionary of weights and biases
    L: number of layers
    keep_prob: probability that a node will be kept
    """
    cache = {'A0': X}

    for i in range(1, L + 1):
        w_key = 'W' + str(i)
        b_key = 'b' + str(i)
        a_prev = cache['A' + str(i - 1)]

        z = np.matmul(weights[w_key], a_prev) + weights[b_key]

        if i < L:
            a = np.tanh(z)
            d = (np.random.rand(a.shape[0], a.shape[1]) < keep_prob).astype(int)
            a = (a * d) / keep_prob
            cache['D' + str(i)] = d
            cache['A' + str(i)] = a
        else:
            t = np.exp(z)
            a = t / np.sum(t, axis=0, keepdims=True)
            cache['A' + str(i)] = a

    return cache
