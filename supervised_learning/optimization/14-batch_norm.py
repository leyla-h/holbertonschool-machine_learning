#!/usr/bin/env python3
"""
Module to create a batch normalization layer in TensorFlow.
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev: The activated output of the previous layer.
        n: The number of nodes in the layer to be created.
        activation: The activation function to use on the output.

    Returns:
        A tensor of the activated output for the layer.
    """
    # Initialize the base Dense layer
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=initializer)

    # Apply the dense layer
    z = dense(prev)

    # Apply Batch Normalization
    # Gamma (scale) is initialized to 1, Beta (offset) to 0 by default
    bn = tf.keras.layers.BatchNormalization(
        axis=-1,
        epsilon=1e-7
    )
    
    # Normalize the output of the dense layer
    z_bn = bn(z)

    # Apply the activation function
    return activation(z_bn)
