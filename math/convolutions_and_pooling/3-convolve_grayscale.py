#!/usr/bin/env python3
"""
Performs a convolution on grayscale images with custom padding and stride.
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w)
    kernel is a numpy.ndarray with shape (kh, kw)
    padding is either a tuple of (ph, pw), 'same', or 'valid'
    stride is a tuple of (sh, sw)
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    oh = (h + (2 * ph) - kh) // sh + 1
    ow = (w + (2 * pw) - kw) // sw + 1

    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = np.sum(
                padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw] *
                kernel, axis=(1, 2)
            )

    return output
