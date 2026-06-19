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

    def __str__(self):
        node_type = "root" if self.is_root else "-> node"
        result = "{} [feature={}, threshold={}]\n".format(
            node_type, self.feature, self.threshold)
        if self.left_child:
            result += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += self.right_child_add_prefix(str(self.right_child))
        return result

    def left_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n" if x else ""
        return new_text

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n" if x else ""
        return new_text

    def max_depth_below(self):
        if self.left_child:
            left = self.left_child.max_depth_below()
        else:
            left = self.depth
        if self.right_child:
            right = self.right_child.max_depth_below()
        else:
            right = self.depth
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        if self.left_child:
            left = self.left_child.count_nodes_below(
                only_leaves=only_leaves)
        else:
            left = 0
        if self.right_child:
            right = self.right_child.count_nodes_below(
                only_leaves=only_leaves)
        else:
            right = 0
        if only_leaves:
            return left + right
        return 1 + left + right

    def get_leaves_below(self):
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves_below()
        if self.right_child:
            leaves += self.right_child.get_leaves_below()
        return leaves

    def update_bounds_below(self):
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

        # left child  -> "greater than" threshold -> lower = threshold
        # right child -> "less or equal" threshold -> upper = threshold
        self.left_child.lower[self.feature] = self.threshold
        self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()


class Leaf:
    """Represents a leaf in a decision tree."""
    def __init__(self, value, depth=None):
        self.value = value
        self.depth = depth
        self.is_leaf = True
        self.sub_population = None

    def __str__(self):
        return "-> leaf [value={}]".format(self.value)

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        pass


class Decision_Tree:
    """Represents a decision tree."""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def __str__(self):
        return self.root.__str__()

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_bounds(self):
        self.root.update_bounds_below()
