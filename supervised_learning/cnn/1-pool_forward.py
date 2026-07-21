#!/usr/bin/env python3
"""
Pooling Forward Propagation
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Output dimensions
    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    # Initialize output volume
    A = np.zeros((m, h_out, w_out, c_prev))

    # Perform pooling
    for i in range(h_out):
        for j in range(w_out):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            slice_A = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(slice_A, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(slice_A, axis=(1, 2))

    return A
