#!/usr/bin/env python3
"""
Module defining the MultiNormal class for a Multivariate Normal distribution.
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initializes the MultiNormal distribution with a data set.

        Parameters:
        data (numpy.ndarray): shape (d, n) containing the data set
            d is the number of dimensions in each data point
            n is the number of data points
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate mean across columns, keeping dimensions to get shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center the data by subtracting the mean vector
        data_centered = data - self.mean

        # Covariance formula for (d, n) shape: (data_centered * data_centered^T) / (n - 1)
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)
