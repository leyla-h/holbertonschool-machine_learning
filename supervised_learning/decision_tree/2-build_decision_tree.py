#!/usr/bin/env python3
"""Module to build and represent a decision tree"""
import numpy as np


class Node:
    """Class representing a node in a decision tree"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes a Node"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculates the maximum depth below this node"""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Counts the number of nodes or leaves below this node"""
        left = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right = self.right_child.count_nodes_below(only_leaves=only_leaves)
        if only_leaves:
            return left + right
        return 1 + left + right

    def __str__(self):
        """String representation of the Node"""
        if self.is_root:
            res = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            res = f"node [feature={self.feature}, threshold={self.threshold}]"

        left = str(self.left_child).split('\n')
        right = str(self.right_child).split('\n')

        res += "\n+---> " + left[0]
        for line in left[1:]:
            res += "\n| " + line
        res += "\n+---> " + right[0]
        for line in right[1:]:
            res += "\n" + line
        return res


class Leaf(Node):
    """Class representing a leaf in a decision tree"""
    def __init__(self, value, depth=None):
        """Initializes a Leaf"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 for a leaf node"""
        return 1

    def __str__(self):
        """String representation of the Leaf"""
        return f"leaf [value={self.value}]"


class Decision_Tree:
    """Class representing a decision tree"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes a Decision_Tree"""
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

    def depth(self):
        """Returns the depth of the decision tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts the nodes in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """String representation of the tree"""
        return self.root.__str__()
