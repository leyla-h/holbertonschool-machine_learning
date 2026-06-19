#!/usr/bin/env python3
"""
Module to calculate precision
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion: a confusion numpy.ndarray of shape (classes, classes)

    Returns:
        A numpy.ndarray of shape (classes,) containing the precision
        of each class.
    """
    # Precision = TP / (TP + FP)
    # TP is the diagonal of the confusion matrix
    # TP + FP is the sum of each column
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)

    return true_positives / predicted_positives
