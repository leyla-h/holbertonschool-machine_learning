#!/usr/bin/env python3
"""
Module to calculate sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion: a confusion numpy.ndarray of shape (classes, classes)

    Returns:
        A numpy.ndarray of shape (classes,) containing the sensitivity
        of each class.
    """
    # Sensitivity = TP / (TP + FN)
    # TP is the diagonal of the confusion matrix
    # TP + FN is the sum of each row
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)

    return true_positives / actual_positives
