#!/usr/bin/env python3
"""This module defines a Decision Tree class and its components."""
import numpy as np


class Node:
    """Represents a node in a decision tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=0, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False
        self.sub_population = None

    def pred(self, x):
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)

    def get_leaves_below(self):
        return self.left_child.get_leaves_below() + self.right_child.get_leaves_below()

    def update_bounds_below(self):
        if self.is_root:
            self.upper = {i: np.inf for i in range(100)}  # Adjust size if needed
            self.lower = {i: -np.inf for i in range(100)}
        for child in [self.left_child, self.right_child]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

        # left  = "> threshold"  -> lower bound is threshold
        # right = "<= threshold" -> upper bound is threshold
        self.left_child.lower[self.feature] = self.threshold
        self.right_child.upper[self.feature] = self.threshold

        self.left_child.update_bounds_below()
        self.right_child.update_bounds_below()

    def update_indicator(self):
        self.left_child.update_indicator()
        self.right_child.update_indicator()

    def max_depth_below(self):
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        return (1 + self.left_child.count_nodes_below(only_leaves=only_leaves) +
                self.right_child.count_nodes_below(only_leaves=only_leaves))


class Leaf:
    """Represents a leaf in a decision tree."""
    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.is_leaf = True
        self.sub_population = None

    def pred(self, x):
        return self.value

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        pass

    def update_indicator(self):
        def is_large_enough(x):
            res = [np.greater(x[:, key], val) for key, val in self.lower.items() if val != -np.inf]
            return np.all(res, axis=0) if res else np.full(x.shape[0], True)

        def is_small_enough(x):
            res = [np.less_equal(x[:, key], val) for key, val in self.upper.items() if val != np.inf]
            return np.all(res, axis=0) if res else np.full(x.shape[0], True)
        self.indicator = lambda x: np.all([is_large_enough(x), is_small_enough(x)], axis=0)

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1


class Decision_Tree:
    """Represents a decision tree."""
    def __init__(self,
