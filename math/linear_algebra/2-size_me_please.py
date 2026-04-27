#!/usr/bin/env python3
"""Defines a function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix and returns it as a list of integers"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
