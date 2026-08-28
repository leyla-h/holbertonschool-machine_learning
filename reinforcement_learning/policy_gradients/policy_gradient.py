#!/usr/bin/env python3
"""Policy function and Monte-Carlo policy gradient"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix

    Args:
        matrix: numpy.ndarray containing the state
        weight: numpy.ndarray containing the weight

    Returns:
        the policy (action probabilities)
    """
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a
    state and a weight matrix

    Args:
        state: matrix representing the current observation
            of the environment
        weight: matrix of random weight

    Returns:
        the action and the gradient (in this order)
    """
    state = state.reshape(1, -1)
    P = policy(state, weight)

    action = np.random.choice(P.shape[1], p=P[0])

    s = P.reshape(-1, 1)
    softmax_grad = np.diagflat(s) - np.dot(s, s.T)
    dsoftmax = softmax_grad[action, :]
    dlog = dsoftmax / P[0, action]

    grad = state.T.dot(dlog[None, :])

    return action, grad
