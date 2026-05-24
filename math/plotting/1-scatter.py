#!/usr/bin/env python3
"""
Module to generate a scatter plot comparing men's height and weight data.
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Plots a multivariate normal distribution of heights and weights.
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    plt.scatter(x, y, c='m')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")
    plt.show()
