#!/usr/bin/env python3
"""Defines a function that adds two 2D matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise and returns a new matrix"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[a + b for a, b in zip(row1, row2)]
            for row1, row2 in zip(mat1, mat2)]
