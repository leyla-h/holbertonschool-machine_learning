#!/usr/bin/env python3
"""
Module to calculate the F1 score
"""
import numpy as np


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Args:
        confusion: a confusion numpy.ndarray of shape (classes, classes)

    Returns:
        A numpy.ndarray of shape (classes,) containing the F1 score
        of each class.
    """
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    s = sensitivity(confusion)
    p = precision(confusion)

    return 2 * ((p * s) / (p + s))
