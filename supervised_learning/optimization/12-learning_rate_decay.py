#!/usr/bin/env python3
"""
Module to create a learning rate decay schedule in TensorFlow.
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow using inverse
    time decay.

    Args:
        alpha: The original learning rate.
        decay_rate: The weight used to determine the rate at which alpha
            will decay.
        decay_step: The number of passes of gradient descent that should
            occur before alpha is decayed further.

    Returns:
        The learning rate decay schedule.
    """
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return lr_schedule
