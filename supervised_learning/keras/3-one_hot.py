#!/usr/bin/env python3
"""
Module to convert a label vector into a one-hot matrix.
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    labels: a numpy array of labels
    classes: the maximum number of classes found in the labels

    Returns: the one-hot matrix
    """
    if classes is None:
        classes = labels.max() + 1
    
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)
    
    return one_hot_matrix
