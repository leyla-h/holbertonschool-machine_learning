#!/usr/bin/env python3
"""Defines a function that initializes the Q-table for a given
FrozenLakeEnv environment.
"""
import numpy as np


def q_init(env):
    """Initializes the Q-table

    Args:
        env: the FrozenLakeEnv instance

    Returns:
        the Q-table as a numpy.ndarray of zeros
    """
    state_space = env.observation_space.n
    action_space = env.action_space.n

    Q = np.zeros((state_space, action_space))

    return Q
