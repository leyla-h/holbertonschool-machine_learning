#!/usr/bin/env python3
"""Module to calculate the sum total"""


def summation_i_squared(n):
    """Calculates sum of i squared from 1 to n
    Args:
        n: stop condition
    Returns:
        integer value or None
    """
    if not isinstance(n, int) or n < 1:
        return None
    res = (n * (n + 1) * (2 * n + 1)) // 6
    return res
