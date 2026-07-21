#!/usr/bin/env python3
"""
Contrast Adjustment
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.
    """
    return tf.image.random_contrast(image, lower=lower, upper=upper)
