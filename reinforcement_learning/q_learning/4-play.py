#!/usr/bin/env python3
"""Defines a function that has a trained agent play an episode of
FrozenLake.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """Has the trained agent play an episode

    Args:
        env: the FrozenLakeEnv instance
        Q: a numpy.ndarray containing the Q-table
        max_steps: the maximum number of steps in the episode

    Returns:
        total_rewards, rendered_outputs
            total_rewards: the total rewards for the episode
            rendered_outputs: a list of rendered outputs representing
                the board state at each step
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = []

    rendered_outputs.append(env.render())

    for step in range(max_steps):
        action = np.argmax(Q[state])

        new_state, reward, terminated, truncated, _ = env.step(action)

        rendered_outputs.append(env.render())

        state = new_state
        total_rewards += reward

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
