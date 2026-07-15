#!/usr/bin/env python3
""" Function that determines if you should stop gradient descent early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    cost: current validation cost
    opt_cost: lowest recorded validation cost
    threshold: threshold for early stopping
    patience: patience count for early stopping
    count: how long the threshold has not been met
    """
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1

    if count >= patience:
        return True, count
    else:
        return False, count
