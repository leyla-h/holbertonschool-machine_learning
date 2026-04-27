#!/usr/bin/env python3
"""Defines a function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Multiplies two 2D matrices and returns a new matrix"""
    # Check if multiplication is possible: col of mat1 == row of mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize new matrix with dimensions: rows of mat1 x cols of mat2
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            # Calculate dot product of mat1 row i and mat2 col j
            dot_product = sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
            row.append(dot_product)
        result.append(row)

    return result
