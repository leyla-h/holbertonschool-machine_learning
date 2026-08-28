#!/usr/bin/env python3
"""Defines a function that performs the Monte Carlo algorithm to
estimate a value function.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                 gamma=0.99):
    """Performs the Monte Carlo algorithm

    Args:
        env: environment instance
        V: a numpy.ndarray of shape (s,) containing the value estimate
        policy: a function that takes in a state and returns the next
            action to take
        episodes: the total number of episodes to train over
        max_steps: the maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate

    Returns:
        V, the updated value estimate
    """
    for episode in range(episodes):
        state, _ = env.reset()
        episode_history = []

        for step in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode_history.append((state, reward))
            state = new_state

            if terminated or truncated:
                break

        episode_history = np.array(episode_history, dtype=int)
        G = 0
        discounted_returns = []

        for state, reward in episode_history[::-1]:
            G = reward + gamma * G
            discounted_returns.append(G)

        discounted_returns = discounted_returns[::-1]

        visited_states = set()
        for i, (state, _) in enumerate(episode_history):
            if state not in visited_states:
                visited_states.add(state)
                V[state] = V[state] + alpha * (
                    discounted_returns[i] - V[state])

    return V
