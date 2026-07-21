#!/usr/bin/env python3
"""
Brightness Adjustment
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.
    """
    return tf.image.random_brightness(image, max_delta=max_delta)
