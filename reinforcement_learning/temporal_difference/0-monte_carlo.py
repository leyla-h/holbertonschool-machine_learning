#!/usr/bin/env python3
"""Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                 alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the
            next action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V, the updated value estimate
    """
    for ep in range(episodes):
        state, _ = env.reset()
        episode = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        episode = np.array(episode, dtype=int)
        G = 0

        for i, (state, reward) in enumerate(reversed(episode)):
            G = gamma * G + reward
            visited_before = state in episode[:episode.shape[0] - i - 1, 0]
            if not visited_before:
                V[state] = V[state] + alpha * (G - V[state])

    return V
