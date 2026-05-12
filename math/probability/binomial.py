#!/usr/bin/env python3
"""
Module containing the Binomial class to represent a binomial distribution.
"""


class Binomial:
    """
    Represents a binomial distribution.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate sample mean and variance
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Estimate p and n
            # Variance = n * p * (1 - p); Mean = n * p
            # p = 1 - (variance / mean)
            p_est = 1 - (variance / mean)
            n_est = round(mean / p_est)

            # Final values based on rounded n
            self.n = int(n_est)
            self.p = float(mean / n_est)
