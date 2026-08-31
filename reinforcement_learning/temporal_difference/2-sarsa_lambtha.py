#!/usr/bin/env python3
"""SARSA(λ) algorithm"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action
    Args:
        Q: numpy.ndarray containing the q-table
        state: current state
        epsilon: epsilon to use for the calculation
    Returns:
        the next action index
    """
    p = np.random.uniform()
    if p < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                   alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                   epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm
    Args:
        env: environment instance
        Q: numpy.ndarray of shape (s,a) containing the Q table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes
    Returns:
        Q, the updated Q table
    """
    initial_epsilon = epsilon
    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros(Q.shape)
        for step in range(max_steps):
            new_state, reward, terminated, truncated, info = env.step(
                action)
            new_action = epsilon_greedy(Q, new_state, epsilon)
            td_error = (reward + gamma * Q[new_state, new_action]
                        - Q[state, action])
            eligibility *= gamma * lambtha
            eligibility[state, action] += 1.0
            Q = Q + alpha * td_error * eligibility
            state = new_state
            action = new_action
            if terminated or truncated:
                break
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * ep))
    return Q
