#!/usr/bin/env python3
"""Full training with Monte-Carlo policy gradient"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements a full training

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor
        show_result: if True, render the environment every
            1000 episodes

    Returns:
        all values of the score (sum of all rewards during
        one episode loop)
    """
    weight = np.random.rand(4, 2)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            if show_result and episode % 1000 == 0:
                env.render()

            action, grad = policy_gradient(state, weight)
            new_state, reward, terminated, truncated, info = env.step(
                action)

            grads.append(grad)
            rewards.append(reward)

            state = new_state

        score = sum(rewards)
        scores.append(score)

        for i, grad in enumerate(grads):
            G = sum(r * (gamma ** r_i) for r_i, r in enumerate(rewards[i:]))
            weight += alpha * grad * G

        print("Episode: {} Score: {}".format(episode, score))

    return scores
