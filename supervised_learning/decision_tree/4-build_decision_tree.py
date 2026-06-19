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

        # Left child: feature <= threshold (updates upper bound)
        self.left_child.upper = self.upper.copy()
        self.left_child.lower = self.lower.copy()
        self.left_child.upper[self.feature] = self.threshold

        # Right child: feature > threshold (updates lower bound)
        self.right_child.upper = self.upper.copy()
        self.right_child.lower = self.lower.copy()
        self.right_child.lower[self.feature] = self.threshold

        # Recurse
        self.left_child.update_bounds_below()
        self.right_child.update_bounds_below()


class Leaf:
    """Represents a leaf in a decision tree."""

    def __init__(self, value, depth=0):
        """Initializes a Leaf instance."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def get_leaves_below(self):
        """Returns the leaf itself in a list."""
        return [self]

    def update_bounds_below(self):
        """Placeholder for leaf node bounds update."""
        pass


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
