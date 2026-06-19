#!/usr/bin/env python3
"""This module defines a Decision Tree class and its components."""
import numpy as np


class Node:
    """Represents an internal (non-leaf) node in a decision tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=0, is_root=False):
        """Initialize a Node with its splitting feature/threshold,
        children, depth, and root flag."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False
        self.sub_population = None

    def __str__(self):
        """Return a human-readable, indented string representation
        of the node and its subtree."""
        node_type = "root" if self.is_root else "-> node"
        result = "{} [feature={}, threshold={}]\n".format(
            node_type, self.feature, self.threshold)
        if self.left_child:
            result += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += self.right_child_add_prefix(str(self.right_child))
        return result

    def left_child_add_prefix(self, text):
        """Add the visual prefix used to display a left child branch
        when printing the tree."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n" if x else ""
        return new_text

    def right_child_add_prefix(self, text):
        """Add the visual prefix used to display a right child branch
        when printing the tree."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n" if x else ""
        return new_text

    def max_depth_below(self):
        """Recursively compute the maximum depth of the subtree
        rooted at this node."""
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
        """Recursively count the nodes in the subtree rooted at this
        node, optionally counting only the leaves."""
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
        """Recursively collect and return all the leaves found in the
        subtree rooted at this node."""
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves_below()
        if self.right_child:
            leaves += self.right_child.get_leaves_below()
        return leaves

    def update_bounds_below(self):
        """Recursively compute and assign the lower and upper bound
        dictionaries of each node/leaf in the subtree, based on the
        feature splits made by each ancestor node."""
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

    def update_indicator(self):
        """Compute the indicator function of this node from its
        lower and upper bound dictionaries, and store it as the
        attribute Node.indicator."""
        def is_large_enough(x):
            """Return a 1D boolean array of size n_individuals,
            True where every feature of the individual is strictly
            greater than the corresponding lower bound."""
            return np.all(np.array(
                [np.greater(x[:, key], self.lower[key])
                 for key in list(self.lower.keys())]), axis=0)

        def is_small_enough(x):
            """Return a 1D boolean array of size n_individuals,
            True where every feature of the individual is less than
            or equal to the corresponding upper bound."""
            return np.all(np.array(
                [np.less_equal(x[:, key], self.upper[key])
                 for key in list(self.upper.keys())]), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)


class Leaf:
    """Represents a leaf (terminal node) in a decision tree."""
    def __init__(self, value, depth=None):
        """Initialize a Leaf with its predicted value and depth."""
        self.value = value
        self.depth = depth
        self.is_leaf = True
        self.sub_population = None

    def __str__(self):
        """Return a human-readable string representation of the
        leaf."""
        return "-> leaf [value={}]".format(self.value)

    def max_depth_below(self):
        """Return the depth of this leaf (base case for the
        recursive depth computation)."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1, since a leaf counts as a single node (base case
        for the recursive node-counting computation)."""
        return 1

    def get_leaves_below(self):
        """Return a list containing only this leaf (base case for
        the recursive leaf-collection computation)."""
        return [self]

    def update_bounds_below(self):
        """Do nothing: a leaf's bounds are already set by its parent
        node during the recursive bound computation."""
        pass

    def update_indicator(self):
        """Compute the indicator function of this leaf from its
        lower and upper bound dictionaries, and store it as the
        attribute Leaf.indicator."""
        def is_large_enough(x):
            """Return a 1D boolean array of size n_individuals,
            True where every feature of the individual is strictly
            greater than the corresponding lower bound."""
            return np.all(np.array(
                [np.greater(x[:, key], self.lower[key])
                 for key in list(self.lower.keys())]), axis=0)

        def is_small_enough(x):
            """Return a 1D boolean array of size n_individuals,
            True where every feature of the individual is less than
            or equal to the corresponding upper bound."""
            return np.all(np.array(
                [np.less_equal(x[:, key], self.upper[key])
                 for key in list(self.upper.keys())]), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)


class Decision_Tree:
    """Represents a full decision tree."""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a Decision_Tree with its hyperparameters and
        root node."""
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
        """Return a human-readable string representation of the
        whole tree."""
        return self.root.__str__()

    def depth(self):
        """Return the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the total number of nodes in the tree, optionally
        counting only the leaves."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Return a list of all the leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Compute and assign the lower/upper bound dictionaries for
        every node and leaf in the tree."""
        self.root.update_bounds_below()
