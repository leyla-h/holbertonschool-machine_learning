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

        # Covariance formula for (d, n) shape:
        # (data_centered * data_centered^T) / (n - 1)
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a specific data point.

        Parameters:
        x (numpy.ndarray): shape (d, 1) containing the data point

        Returns:
        float: the value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if len(x.shape) != 2 or x.shape[0] != d or x.shape[1] != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        # Vector difference between target point and distribution mean
        diff = x - self.mean

        # Compute exponent component: -0.5 * (x - mu)^T * Sigma^-1 * (x - mu)
        exponent = -0.5 * np.matmul(np.matmul(diff.T, inv), diff)

        # Compute normalization factor: 1 / sqrt((2 * pi)^d * det(Sigma))
        norm_factor = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)

        # The result of exponentiation yields a 1x1 matrix; extract scalar
        pdf_val = norm_factor * np.exp(exponent[0, 0])

        return pdf_val
