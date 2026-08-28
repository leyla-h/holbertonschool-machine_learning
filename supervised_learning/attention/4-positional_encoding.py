#!/usr/bin/env python3
"""Positional Encoding module for transformers"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer

    max_seq_len is an integer representing the maximum sequence length
    dm is the model depth

    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors
    """
    PE = np.zeros((max_seq_len, dm))
    positions = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.power(10000, (2 * (np.arange(dm) // 2)) / np.float32(dm))

    angles = positions / div_term

    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    return PE
