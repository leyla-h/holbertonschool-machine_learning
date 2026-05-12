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
        # Convert k to an integer as required
        k = int(k)

        # Poisson distribution is defined for k >= 0
        if k < 0:
            return 0

        # Mathematical constant e
        e = 2.7182818285

        # Calculate factorial of k
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # PMF Formula: (e^-λ * λ^k) / k!
        pmf_val = ((e ** -self.lambtha) * (self.lambtha ** k)) / factorial

        return pmf_val
