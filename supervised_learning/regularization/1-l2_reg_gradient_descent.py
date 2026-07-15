#!/usr/bin/env python3
"""Gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization

    Y is a one-hot numpy.ndarray of shape (classes, m)
    weights is a dictionary of the weights and biases of the network
    cache is a dictionary of the outputs of each layer of the network
    alpha is the learning rate
    lambtha is the L2 regularization parameter
    L is the number of layers of the network

    The neural network uses tanh activations on each layer except the
    last, which uses a softmax activation. Weights/biases are updated
    in place.
    """
    m = Y.shape[1]
    weights_copy = weights.copy()

    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights_copy['W' + str(i)]
        b = weights_copy['b' + str(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)

        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db
