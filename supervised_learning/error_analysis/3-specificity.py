#!/usr/bin/env python3
"""
Module to calculate specificity
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion: a confusion numpy.ndarray of shape (classes, classes)

    Returns:
        A numpy.ndarray of shape (classes,) containing the specificity
        of each class.
    """
    tp = np.diag(confusion)
    row_sum = np.sum(confusion, axis=1)
    col_sum = np.sum(confusion, axis=0)
    total_sum = np.sum(confusion)

    fp = col_sum - tp
    tn = total_sum - row_sum - fp

    return tn / (tn + fp)
