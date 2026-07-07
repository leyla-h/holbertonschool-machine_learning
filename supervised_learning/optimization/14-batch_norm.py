#!/usr/bin/env python3
"""
Module to create a batch normalization layer in TensorFlow.
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.
    """
    # 1. Base Dense layer with specified initializer
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=initializer)

    # 2. Linear transformation
    z = dense(prev)

    # 3. Batch Normalization
    # Gamma (scale) default 'ones', Beta (offset) default 'zeros'
    bn = tf.keras.layers.BatchNormalization(
        axis=-1,
        epsilon=1e-7,
        gamma_initializer='ones',
        beta_initializer='zeros'
    )
    
    # 4. Normalize
    z_bn = bn(z)

    # 5. Activation
    return activation(z_bn)
