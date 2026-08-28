#!/usr/bin/env python3
"""Defines a function that uses epsilon-greedy to determine the next
action.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Uses epsilon-greedy to determine the next action

    Args:
        Q: a numpy.ndarray containing the q-table
        state: the current state
        epsilon: the epsilon to use for the calculation

    Returns:
        the next action index
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])

    return action
