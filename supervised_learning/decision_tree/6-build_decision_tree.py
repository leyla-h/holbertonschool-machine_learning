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

    def pred(self, x):
        """Returns the leaf value for a single sample x via tree traversal."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

    def get_leaves_below(self):
        """Returns the list of all leaves below this node."""
        return self.left_child.get_leaves_below() + \
            self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """Recursively updates bounds."""
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
        """Recursively updates indicators."""
        self.left_child.update_indicator()
        self.right_child.update_indicator()

    def __str__(self):
        """String representation of the tree."""
        # Note: Ensure this matches your earlier requirement
        return ""


class Leaf:
    """Represents a leaf in a decision tree."""

    def __init__(self, value, depth=0):
        """Initializes a Leaf instance."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def pred(self, x):
        """Returns the value of the leaf."""
        return self.value

    def get_leaves_below(self):
        """Returns the leaf itself."""
        return [self]

    def update_indicator(self):
        """Computes the indicator function for this leaf."""
        def is_large_enough(x):
            res = [np.greater(x[:, key], val) for key, val in self.lower.items()]
            return np.all(res, axis=0) if res else np.full(x.shape[0], True)

        def is_small_enough(x):
            res = [np.less_equal(x[:, key], val) for key, val in self.upper.items()]
            return np.all(res, axis=0) if res else np.full(x.shape[0], True)

        self.indicator = lambda x: np.all([is_large_enough(x), is_small_enough(x)], axis=0)


class Decision_Tree:
    """Represents a decision tree."""

    def __init__(self, root=None):
        """Initializes a Decision_Tree instance."""
        self.root = root

    def pred(self, x):
        """Traverses the tree to predict for x."""
        return self.root.pred(x)

    def get_leaves(self):
        """Returns the list of all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Updates the bounds for all nodes."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Updates indicators and sets up the vectorized predict function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            """Vectorized prediction."""
            # Matrix of shape (n_leaves, n_individuals)
            indicators = np.array([leaf.indicator(A) for leaf in leaves])
            # Array of values associated with each leaf
            values = np.array([leaf.value for leaf in leaves])
            # For each column (individual), find the index where indicator is True
            leaf_indices = np.argmax(indicators, axis=0)
            return values[leaf_indices]

        self.predict = predict
