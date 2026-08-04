#!/usr/bin/env python3
"""
Module for performing Principal Component Analysis (PCA).
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    
    Parameters:
    - X: numpy.ndarray of shape (n, d) where n is the number of data points
      and d is the number of dimensions.
    - var: float representing the fraction of the variance that the PCA
      transformation should maintain.
      
    Returns:
    - W: numpy.ndarray of shape (d, nd) representing the weights matrix.
    """
    # Perform Singular Value Decomposition (SVD) on X
    _, S, Vt = np.linalg.svd(X)
    
    # Calculate the variance explained by each component
    # Variance is proportional to the square of the singular values
    explained_variance = (S ** 2) / np.sum(S ** 2)
    
    # Calculate the cumulative variance
    cumulative_variance = np.cumsum(explained_variance)
    
    # Find the number of dimensions (nd) that maintain at least 'var' fraction of variance
    # We look for the first index where cumulative variance >= var
    nd = np.argmax(cumulative_variance >= var) + 1
    
    # Vt has shape (d, d), where rows are the principal axes.
    # We need the first nd rows, and we transpose to get shape (d, nd).
    W = Vt.T[:, :nd]
    
    return W
