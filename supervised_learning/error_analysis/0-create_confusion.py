#!/usr/bin/env python3
"""
Module to create a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes)
        logits: one-hot numpy.ndarray of shape (m, classes)

    Returns:
        A confusion numpy.ndarray of shape (classes, classes)
    """
    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    for i in range(len(true_labels)):
        confusion[true_labels[i], pred_labels[i]] += 1

    return confusion
