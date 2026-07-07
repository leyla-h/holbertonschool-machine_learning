#!/usr/bin/env python3
"""
Module to calculate the weighted moving average of a data set.
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    Args:
        data: The list of data to calculate the moving average of.
        beta: The weight used for the moving average.

    Returns:
        A list containing the moving averages of data.
    """
    moving_averages = []
    v = 0
    for t, theta in enumerate(data, 1):
        v = beta * v + (1 - beta) * theta
        v_corrected = v / (1 - beta ** t)
        moving_averages.append(v_corrected)
    return moving_averages
