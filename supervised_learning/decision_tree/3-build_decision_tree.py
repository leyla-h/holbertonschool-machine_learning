#!/usr/bin/env python3
"""This module defines a Decision Tree class and its components."""


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


class Leaf:
    """Represents a leaf in a decision tree."""

    def __init__(self, value, depth=0):
        """Initializes a Leaf instance."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        """Returns a string representation of the leaf."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Returns the leaf itself in a list."""
        return [self]


class Decision_Tree:
    """Represents a decision tree."""

    def __init__(self, root=None):
        """Initializes a Decision_Tree instance."""
        self.root = root

    def get_leaves(self):
        """Returns the list of all leaves of the tree."""
        return self.root.get_leaves_below()
