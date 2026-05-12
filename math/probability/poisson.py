#!/usr/bin/env python3
"""
Module containing the Poisson class to represent a Poisson distribution.
"""


class Poisson:
    """
    Represents a poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution.

        Args:
            data (list): List of data to estimate the distribution.
            lambtha (float): Expected number of occurrences.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes".

        Args:
            k (int): The number of successes.

        Returns:
            float: The PMF value for k.
        """
        k = int(k)
        if k < 0:
            return 0

        e = 2.7182818285
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        pmf_val = ((e ** -self.lambtha) * (self.lambtha ** k)) / factorial
        return pmf_val

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of "successes".

        Args:
            k (int): The number of successes.

        Returns:
            float: The CDF value for k.
        """
        k = int(k)
        if k < 0:
            return 0

        cdf_val = 0
        for i in range(k + 1):
            cdf_val += self.pmf(i)

        return cdf_val
