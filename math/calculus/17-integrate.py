#!/usr/bin/env python3
"""Module to calculate integral"""


def poly_integral(poly, C=0):
    """Calculates integral"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None
    res = [C]
    for i in range(len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
        val = poly[i] / (i + 1)
        if val % 1 == 0:
            res.append(int(val))
        else:
            res.append(val)
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    return res
