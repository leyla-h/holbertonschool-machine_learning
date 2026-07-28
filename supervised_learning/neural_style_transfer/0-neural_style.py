#!/usr/bin/env python3
"""
Initialize NST class for Neural Style Transfer.
"""

import tensorflow as tf
import numpy as np


class NST:
    """
    Performs Neural Style Transfer.
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for NST.
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float, np.number)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float, np.number)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = float(alpha) if isinstance(alpha, float) or not isinstance(alpha, int) else (alpha if alpha != int(alpha) else int(alpha))
        # Keep original int format if it's an int, or float if float
        if isinstance(alpha, int) and not isinstance(alpha, bool):
            self.alpha = alpha
        else:
            self.alpha = float(alpha)

        if isinstance(beta, int) and not isinstance(beta, bool):
            self.beta = beta
        else:
            self.beta = float(beta)

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels.
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        max_dim = max(h, w)
        scale = 512 / max_dim

        h_new = int(round(h * scale))
        w_new = int(round(w * scale))

        image_resized = tf.image.resize(
            image,
            [h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )

        image_scaled = image_resized / 255.0
        image_scaled = tf.clip_by_value(image_scaled, 0.0, 1.0)
        image_expanded = tf.expand_dims(image_scaled, axis=0)

        return image_expanded
