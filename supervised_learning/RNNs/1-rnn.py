#!/usr/bin/env python3
"""Defines the rnn function"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN

    rnn_cell -- instance of RNNCell used for the forward propagation
    X         -- numpy.ndarray of shape (t, m, i) containing the data
                to be used
                t is the maximum number of time steps
                m is the batch size
                i is the dimensionality of the data
    h_0       -- numpy.ndarray of shape (m, h) containing the initial
                hidden state

    Returns: H, Y
    H -- numpy.ndarray containing all of the hidden states
    Y -- numpy.ndarray containing all of the outputs
    """
    t = X.shape[0]
    m, h = h_0.shape

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    Y = []
    h_prev = h_0
    for step in range(t):
        h_next, y = rnn_cell.forward(h_prev, X[step])
        H[step + 1] = h_next
        Y.append(y)
        h_prev = h_next

    Y = np.array(Y)

    return H, Y
