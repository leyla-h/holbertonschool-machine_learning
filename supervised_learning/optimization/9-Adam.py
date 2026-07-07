#!/usr/bin/env python3
"""
Module to update a variable using the Adam optimization algorithm.
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.

    Args:
        alpha: The learning rate.
        beta1: The weight used for the first moment.
        beta2: The weight used for the second moment.
        epsilon: A small number to avoid division by zero.
        var: The variable to be updated.
        grad: The gradient of var.
        v: The previous first moment of var.
        s: The previous second moment of var.
        t: The time step used for bias correction.

    Returns:
        The updated variable, the new first moment, and the new second moment.
    """
    # 1. Update biased first moment estimate
    v_new = beta1 * v + (1 - beta1) * grad
    
    # 2. Update biased second raw moment estimate
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)
    
    # 3. Compute bias-corrected first moment estimate
    v_corrected = v_new / (1 - beta1 ** t)
    
    # 4. Compute bias-corrected second raw moment estimate
    s_corrected = s_new / (1 - beta2 ** t)
    
    # 5. Update the variable
    var_new = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))
    
    return var_new, v_new, s_new
