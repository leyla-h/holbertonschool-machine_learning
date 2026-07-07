#!/usr/bin/env python3
"""
Module to calculate learning rate decay using inverse time decay.
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay.

    Args:
        alpha: The original learning rate.
        decay_rate: The weight used to determine the rate at which alpha
            decays.
        global_step: The number of passes of gradient descent that have
            elapsed.
        decay_step: The number of passes that should occur before alpha is
            decayed further.

    Returns:
        The updated value for alpha.
    """
    # Calculate how many times the decay has been applied
    decay_count = global_step // decay_step

    # Calculate the new learning rate
    alpha_updated = alpha / (1 + decay_rate * decay_count)

    return alpha_updated
