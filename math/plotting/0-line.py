#!/usr/bin/env python3
"""
Module to plot a mathematical line graph using matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots a cubic function as a solid red line with defined x-axis limits.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)
    plt.plot(x, y, color='red', linestyle='solid')
    plt.xlim(0, 10)
    plt.show()
