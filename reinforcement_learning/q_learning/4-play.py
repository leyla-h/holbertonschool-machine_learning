#!/usr/bin/env python3
"""Has the trained agent play an episode"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode

    env: the FrozenLakeEnv instance
    Q: numpy.ndarray containing the Q-table
    max_steps: maximum number of steps in the episode

    Returns: total_rewards, rendered_outputs
        total_rewards: the total rewards for the episode
        rendered_outputs: a list of rendered outputs representing
                           the board state at each step
    """
    state, _ = env.reset()
    rendered_outputs = [env.render()]
    total_rewards = 0

    for step in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, terminated, truncated, info = env.step(action)
        rendered_outputs.append(env.render())
        total_rewards += reward
        state = new_state

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
