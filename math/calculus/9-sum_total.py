#!/usr/bin/env python3
"""Module to calculate the sum of i squared"""


def summation_i_squared(n):
    """Calculates the sum of i^2 from 1 to n using the square pyramidal formula
    Args:
        n: The stopping condition
    Returns:
        The integer value of the sum, or None if n is invalid
    """
    if not isinstance(n, int) or n < 1:
        return None
    # Formula for sum of squares: n(n+1)(2n+1) / 6
    return (n * (n + 1) * (2 * n + 1)) // 6
