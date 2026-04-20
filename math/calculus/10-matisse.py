#!/usr/bin/env python3
"""Module to calculate the derivative of a polynomial"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for x in poly:
        if not isinstance(x, (int, float)):
            return None

    if len(poly) == 1:
        return [0]

    deriv = []
    for i in range(1, len(poly)):
        deriv.append(poly[i] * i)

    return deriv
