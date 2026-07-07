#!/usr/bin/env python3
"""
Module to update a variable using the RMSProp optimization algorithm.
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using RMSProp.

    Args:
        alpha: The learning rate.
        beta2: The RMSProp weight.
        epsilon: A small number to avoid division by zero.
        var: The variable to be updated.
        grad: The gradient of var.
        s: The previous second moment of var.

    Returns:
        The updated variable and the new moment, respectively.
    """
    # Calculate the new second moment
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # Update the variable
    var_new = var - alpha * (grad / (np.sqrt(s_new) + epsilon))

    return var_new, s_new
