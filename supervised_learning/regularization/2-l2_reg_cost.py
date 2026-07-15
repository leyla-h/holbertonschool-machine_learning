#!/usr/bin/env python3
""" Function that calculates the cost of a neural network with L2
regularization
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    cost: tensor containing the cost of the network without L2 regularization
    model: Keras model that includes layers with L2 regularization
    Returns: a tensor containing the total cost for each layer of the network
    """
    l2_losses = model.losses
    return cost + tf.stack(l2_losses)
