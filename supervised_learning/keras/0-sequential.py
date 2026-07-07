#!/usr/bin/env python3
"""
Module to build a sequential neural network using Keras.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    nx: number of input features
    layers: list containing the number of nodes in each layer
    activations: list containing the activation functions for each layer
    lambtha: L2 regularization parameter
    keep_prob: probability that a node will be kept for dropout

    Returns: the keras model
    """
    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            layer = K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer,
                input_shape=(nx,)
            )
        else:
            layer = K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer
            )

        model.add(layer)

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
