#!/usr/bin/env python3
"""Defines a function that transposes a 2D matrix"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
