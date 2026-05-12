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

            # Calculate lambtha (mean of the data)
            self.lambtha = float(sum(data) / len(data))
