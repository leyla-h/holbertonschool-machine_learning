#!/usr/bin/env python3
"""
Module to update a variable using gradient descent with momentum.
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum.

    Args:
        alpha: The learning rate.
        beta1: The momentum weight.
        var: The variable to be updated.
        grad: The gradient of var.
        v: The previous first moment of var.

    Returns:
        The updated variable and the new moment, respectively.
    """
    # Calculate the new velocity (first moment)
    v_new = beta1 * v + (1 - beta1) * grad

    # Update the variable
    var_new = var - alpha * v_new

    return var_new, v_new
