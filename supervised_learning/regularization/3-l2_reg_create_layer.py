#!/usr/bin/env python3
""" Function that creates a neural network layer in TensorFlow with L2
regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    prev: tensor containing the output of the previous layer
    n: number of nodes in the new layer
    activation: activation function for the layer
    lambtha: L2 regularization parameter
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode='fan_avg')
    regularizer = tf.keras.regularizers.l2(lambtha)
    layer = tf.keras.layers.Dense(n,
                                  activation=activation,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer)
    return layer(prev)
