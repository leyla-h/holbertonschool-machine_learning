#!/usr/bin/env python3
""" Function that creates a layer of a neural network using dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    prev: tensor containing the output of the previous layer
    n: number of nodes the new layer should contain
    activation: activation function for the new layer
    keep_prob: probability that a node will be kept
    training: boolean indicating whether the model is in training mode
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode='fan_avg')
    layer = tf.keras.layers.Dense(n,
                                  activation=activation,
                                  kernel_initializer=initializer)
    dropout = tf.keras.layers.Dropout(1 - keep_prob)
    output = layer(prev)
    return dropout(output, training=training)
