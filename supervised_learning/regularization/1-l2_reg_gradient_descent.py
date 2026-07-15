#!/usr/bin/env python3
""" Function that updates the weights and biases of a neural network using
gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Y: one-hot numpy.ndarray of shape (classes, m)
    weights: dictionary of the weights and biases
    cache: dictionary of the outputs of each layer
    alpha: learning rate
    lambtha: L2 regularization parameter
    L: number of layers
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        w_key = 'W' + str(i)
        b_key = 'b' + str(i)
        a_prev = cache['A' + str(i - 1)]

        dw = (1 / m) * np.matmul(dz, a_prev.T)
        dw += (lambtha / m) * weights[w_key]
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        weights[w_key] -= alpha * dw
        weights[b_key] -= alpha * db

        if i > 1:
            dz = np.matmul(weights[w_key].T, dz) * (1 - np.square(a_prev))
