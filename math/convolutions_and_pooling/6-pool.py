#!/usr/bin/env python3
"""
Performs pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    images: numpy.ndarray with shape (m, h, w, c)
    kernel_shape: tuple of (kh, kw)
    stride: tuple of (sh, sw)
    mode: 'max' or 'avg'
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1

    output = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2)
                )
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2)
                )

    return output
