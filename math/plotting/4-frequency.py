#!/usr/bin/env python3
"""
Module to plot a histogram of student grades.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram tracking the distribution of student scores.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')

    plt.xlim(0, 100)
    plt.xticks(bins)

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    plt.show()
