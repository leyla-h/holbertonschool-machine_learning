#!/usr/bin/env python3
"""
Convolutional Forward Propagation
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over
    a convolutional layer of a neural network.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        return

    # Output dimensions
    h_out = int((h_prev + 2 * ph - kh) / sh) + 1
    w_out = int((w_prev + 2 * pw - kw) / sw) + 1

    # Pad A_prev
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Initialize output volume
    Z = np.zeros((m, h_out, w_out, c_new))

    # Perform convolution
    for i in range(h_out):
        for j in range(w_out):
            # Find the corners of the current slice
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            # Define the slice
            slice_A = A_prev_pad[
                :, vert_start:vert_end, horiz_start:horiz_end, :
            ]

            # Compute the convolution for each channel
            for k in range(c_new):
                Z[:, i, j, k] = np.sum(
                    slice_A * W[:, :, :, k],
                    axis=(1, 2, 3)
                )

    # Add bias
    Z = Z + b

    # Apply activation function
    A = activation(Z)

    return A
