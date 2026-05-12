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

            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period.
        """
        if x < 0:
            return 0

        e = 2.7182818285
        pdf_val = self.lambtha * (e ** (-self.lambtha * x))

        return pdf_val

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period.
        """
        if x < 0:
            return 0

        e = 2.7182818285
        # Formula: 1 - e^(-λ * x)
        cdf_val = 1 - (e ** (-self.lambtha * x))

        return cdf_val
