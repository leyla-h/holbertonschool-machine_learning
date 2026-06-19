#!/usr/bin/env python3
"""This module defines a Decision Tree class and its components."""


def left_child_add_prefix(text):
    """Adds prefix for left child."""
    lines = text.split("\n")
    new_text = "+---> " + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("| " + x) + "\n"
    return new_text


def right_child_add_prefix(text):
    """Adds prefix for right child."""
    lines = text.split("\n")
    new_text = "+---> " + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("  " + x) + "\n"
    return new_text


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

    def __str__(self):
        """Returns a string representation of the node."""
        if self.is_root:
            res = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            res = f"node [feature={self.feature}, threshold={self.threshold}]\n"
        res += left_child_add_prefix(str(self.left_child))
        res += right_child_add_prefix(str(self.right_child))
        return res


class Leaf:
    """Represents a leaf in a decision tree."""

    def __init__(self, value, depth=0):
        """Initializes a Leaf instance."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        """Returns a string representation of the leaf."""
        return f"leaf [value={self.value}]"


class Decision_Tree:
    """Represents a decision tree."""

    def __init__(self, root=None):
        """Initializes a Decision_Tree instance."""
        self.root = root

    def __str__(self):
        """Returns a string representation of the decision tree."""
        return self.root.__str__()
