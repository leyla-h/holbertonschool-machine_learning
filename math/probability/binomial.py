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

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p_est = 1 - (variance / mean)
            n_est = round(mean / p_est)

            self.n = int(n_est)
            self.p = float(mean / n_est)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes".
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        def factorial(n):
            """Helper to calculate factorial"""
            res = 1
            for i in range(1, n + 1):
                res *= i
            return res

        n_fact = factorial(self.n)
        k_fact = factorial(k)
        nk_fact = factorial(self.n - k)
        combination = n_fact / (k_fact * nk_fact)

        pmf_val = combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))

        return pmf_val

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of "successes".
        """
        k = int(k)
        if k < 0:
            return 0

        cdf_val = 0
        for i in range(k + 1):
            cdf_val += self.pmf(i)

        return cdf_val
