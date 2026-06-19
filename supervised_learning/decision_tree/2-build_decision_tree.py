#!/usr/bin/env python3
"""Decision Tree module with Node, Leaf, and Decision_Tree classes."""


class Node:
    """Represents an internal node in a decision tree."""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 is_root=False, depth=0):
        """Initialize a Node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_root = is_root
        self.depth = depth
        self.is_leaf = False

    def left_child_add_prefix(self, text):
        """Add prefix formatting for the left child."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix formatting for the right child."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return string representation of the node and its subtree."""
        if self.is_root:
            node_str = (
                f"root [feature={self.feature}, threshold={self.threshold}]"
            )
        else:
            node_str = (
                f"-> node [feature={self.feature},"
                f" threshold={self.threshold}]"
            )

        left_str = self.left_child_add_prefix(self.left_child.__str__())
        right_str = self.right_child_add_prefix(self.right_child.__str__())

        return node_str + "\n" + left_str + right_str.rstrip("\n")


class Leaf:
    """Represents a leaf node in a decision tree."""

    def __init__(self, value, depth=0):
        """Initialize a Leaf."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        """Return string representation of the leaf."""
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Represents a decision tree."""

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """Initialize a Decision_Tree."""
        self.rng = None
        self.root = root
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.split_criterion = split_criterion
        self.predict = None

    def __str__(self):
        """Return string representation of the decision tree."""
        return self.root.__str__()
