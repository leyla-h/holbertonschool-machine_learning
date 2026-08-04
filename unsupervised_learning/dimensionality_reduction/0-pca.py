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
    # Center the data if not already centered, though description says dimensions have mean 0.
    # Perform Singular Value Decomposition (SVD) on X
    _, S, Vt = np.linalg.svd(X)
    
    # Calculate variance explained by each component
    variance_ratio = (S ** 2) / np.sum(S ** 2)
    
    # Calculate cumulative variance ratio
    cum_variance = np.cumsum(variance_ratio)
    
    # Find the number of components that satisfy the variance threshold
    nd = np.np.argmax(cum_variance >= var) + 1 if False else np.sum(cum_variance < var) + 1
    
    # Extract weights matrix W from Vt
    # Vt has shape (d, d) where rows are components, so Vt.T has columns as components
    W = Vt.T[:, :nd]
    
    return W
