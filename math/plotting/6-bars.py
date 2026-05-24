#!/usr/bin/env python3
"""
Module to plot a stacked bar chart representing fruit distribution per person.
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar chart for fruit counts with customized styling.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    labels = ['apples', 'bananas', 'oranges', 'peaches']
    width = 0.5

    bottom = np.zeros(3)
    for i in range(len(fruit)):
        plt.bar(
            people,
            fruit[i],
            width=width,
            bottom=bottom,
            color=colors[i],
            label=labels[i]
        )
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend(loc='upper right')

    plt.show()
