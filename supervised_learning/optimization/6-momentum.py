#!/usr/bin/env python3
"""
Module to create an optimizer using gradient descent with momentum in TF.
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm.

    Args:
        alpha: The learning rate.
        beta1: The momentum weight.

    Returns:
        The initialized optimizer.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
