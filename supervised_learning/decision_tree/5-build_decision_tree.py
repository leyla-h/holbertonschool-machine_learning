#!/usr/bin/env python3
"""This module defines a Decision Tree class and its components."""
import numpy as np


class Node:
    """Represents a node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=0, is_root=False):
        """Initializes a Node instance."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def get_leaves_below(self):
        """Returns the list of all leaves below this node."""
        return self.left_child.get_leaves_below() + \
            self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """Recursively updates the lower and upper bounds of children."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

        self.left_child.upper[self.feature] = self.threshold
        self.right_child.lower[self.feature] = self.threshold

        self.left_child.update_bounds_below()
        self.right_child.update_bounds_below()

    def update_indicator(self):
        """Recursively updates the indicator function for all children."""
        self.left_child.update_indicator()
        self.right_child.update_indicator()


class Leaf:
    """Represents a leaf in a decision tree."""

    def __init__(self, value, depth=0):
        """Initializes a Leaf instance."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def get_leaves_below(self):
        """Returns the list of the leaf itself."""
        return [self]

    def update_bounds_below(self):
        """Leaf nodes do not have children to update."""
        pass

    def update_indicator(self):
        """Computes the indicator function for this leaf."""
        def is_large_enough(x):
            """Returns True if features > lower bounds."""
            res = [np.greater(x[:, key], val) for key, val in self.lower.items()]
            return np.all(res, axis=0)

        def is_small_enough(x):
            """Returns True if features <= upper bounds."""
            res = [np.less_equal(x[:, key], val) for key, val in self.upper.items()]
            return np.all(res, axis=0)

        self.indicator = lambda x: np.all(
            [is_large_enough(x), is_small_enough(x)], axis=0
        )


class Decision_Tree:
    """Represents a decision tree."""

    def __init__(self, root=None):
        """Initializes a Decision_Tree instance."""
        self.root = root

    def get_leaves(self):
        """Returns the list of all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Updates the bounds for all nodes in the tree."""
        self.root.update_bounds_below()

    def update_indicator(self):
        """Updates the indicator functions for all leaves."""
        self.root.update_indicator()
