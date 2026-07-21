#!/usr/bin/env python3
"""
Identity Block for Deep Residual Learning
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    Deep Residual Learning for Image Recognition (2015).
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # First component of main path (1x1 conv)
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=init
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # Second component of main path (3x3 conv)
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=init
    )(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # Third component of main path (1x1 conv)
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=init
    )(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Final step: Add shortcut value to main path, and pass it through a ReLU activation
    add = K.layers.Add()([bn3, A_prev])
    output = K.layers.Activation('relu')(add)

    return output
