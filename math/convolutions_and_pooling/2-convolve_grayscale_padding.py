#!/usr/bin/env python3
"""
Performs a convolution on grayscale images with custom padding.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
    the convolution
    padding is a tuple of (ph, pw)
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    oh = h + (2 * ph) - kh + 1
    ow = w + (2 * pw) - kw + 1

    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = np.sum(
                padded_images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2)
            )

    return output
