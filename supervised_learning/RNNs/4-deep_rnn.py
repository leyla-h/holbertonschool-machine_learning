#!/usr/bin/env python3
"""Defines the deep_rnn function"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN

    rnn_cells -- list of length l of RNNCell instances used for the
                forward propagation
                l is the number of layers
    X          -- numpy.ndarray of shape (t, m, i) containing the data
                to be used
                t is the maximum number of time steps
                m is the batch size
                i is the dimensionality of the data
    h_0        -- numpy.ndarray of shape (l, m, h) containing the
                initial hidden state
                h is the dimensionality of the hidden state

    Returns: H, Y
    H -- numpy.ndarray containing all of the hidden states
    Y -- numpy.ndarray containing all of the outputs
    """
    layers = len(rnn_cells)
    t, m, i = X.shape
    _, _, h = h_0.shape

    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0

    Y = []
    for step in range(t):
        x = X[step]
        for layer in range(layers):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x)
            H[step + 1, layer] = h_next
            x = h_next
        Y.append(y)

    Y = np.array(Y)

    return H, Y
