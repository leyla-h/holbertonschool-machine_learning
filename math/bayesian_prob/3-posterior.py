#!/usr/bin/env python3
"""
Module to calculate the posterior probability of obtaining data.
"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for various hypothetical probabilities.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        err = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate Likelihood: nCr * (P^x) * ((1-P)^(n-x))
    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_nx = np.math.factorial(n - x)
    combination = fact_n / (fact_x * fact_nx)

    l_hood = combination * (P ** x) * ((1 - P) ** (n - x))

    # Intersection = Likelihood * Prior
    intersect = l_hood * Pr

    # Marginal Probability = Sum of Intersections
    marginal_prob = np.sum(intersect)

    # Posterior = Intersection / Marginal
    return intersect / marginal_prob
