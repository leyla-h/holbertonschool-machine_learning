#!/usr/bin/env python3
"""
Module to calculate the likelihood of obtaining data for various probabilities.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining x successes in n trials.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # nCr calculation: n! / (x! * (n - x)!)
    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_nx = np.math.factorial(n - x)
    combination = fact_n / (fact_x * fact_nx)

    # Binomial likelihood: (nCr) * (P^x) * ((1-P)^(n-x))
    # This operation is vectorized over the numpy array P
    l_hood = combination * (P ** x) * ((1 - P) ** (n - x))

    return l_hood
