#!/usr/bin/env python3
""" Function that calculates the cost of a neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    cost: cost of the network without L2 regularization
    lambtha: regularization parameter
    weights: dictionary of weights and biases
    L: number of layers
    m: number of data points
    """
    sum_squared_weights = 0

    for i in range(1, L + 1):
        w_key = 'W' + str(i)
        sum_squared_weights += np.sum(np.square(weights[w_key]))

    l2_term = (lambtha / (2 * m)) * sum_squared_weights

    return cost + l2_term
