#!/usr/bin/env python3
"""Defines the GRUCell class"""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """Class constructor

        i -- dimensionality of the data
        h -- dimensionality of the hidden state
        o -- dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step

        h_prev -- numpy.ndarray of shape (m, h) containing the previous
                  hidden state
        x_t     -- numpy.ndarray of shape (m, i) containing the data
                  input for the cell

        Returns: h_next, y
        h_next -- the next hidden state
        y       -- the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z_t = self._sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r_t = self._sigmoid(np.matmul(concat, self.Wr) + self.br)

        concat_r = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_r, self.Wh) + self.bh)

        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(
            np.exp(y_linear), axis=1, keepdims=True)

        return h_next, y

    @staticmethod
    def _sigmoid(x):
        """Applies the sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
