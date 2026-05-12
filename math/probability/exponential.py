#!/usr/bin/env python3
"""
Module containing the Exponential class to represent
an exponential distribution.
"""


class Exponential:
    """
    Represents an exponential distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Exponential distribution.

        Args:
            data (list): List of data to estimate the distribution.
            lambtha (float): Expected number of occurrences.
        """
        if data is None:
            # Check lambtha if data is not provided
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Validate data if provided
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate lambtha (inverse of the mean of data)
            self.lambtha = float(1 / (sum(data) / len(data)))
