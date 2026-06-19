#!/usr/bin/env python3
"""
This module defines the Node, Leaf and Decision_Tree classes, the
building blocks used to construct, print, and (eventually) train and
use a decision tree.
"""
import numpy as np


class Node:
    """
    Represents an internal (non-leaf) node of a decision tree.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initializes a Node instance.

        Args:
            feature: index of the feature used to split the node.
            threshold: threshold value used to split the node.
            left_child (Node): the left child of this node.
            right_child (Node): the right child of this node.
            is_root (bool): whether this node is the root of the tree.
            depth (int): the depth of this node within the tree.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def __str__(self):
        """
        Builds and returns a human readable, indented string
        representation of the subtree rooted at this node.

        Returns:
            str: the string representation of the subtree.
        """
        if self.is_root:
            s = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            s = f"-> node [feature={self.feature}, "
            s += f"threshold={self.threshold}]"

        result = s + "\n"
        if self.left_child is not None:
            result += self.left_child_add_prefix(str(self.left_child))
        if self.right_child is not None:
            result += self.right_child_add_prefix(str(self.right_child))
        return result

    def left_child_add_prefix(self, text):
        """
        Indents and prefixes the string representation of a left child
        so that it lines up correctly underneath its parent node.

        Args:
            text (str): the string representation of the left child.

        Returns:
            str: the indented, prefixed string.
        """
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Indents and prefixes the string representation of a right child
        so that it lines up correctly underneath its parent node.

        Args:
            text (str): the string representation of the right child.

        Returns:
            str: the indented, prefixed string.
        """
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text


class Leaf(Node):
    """
    Represents a leaf of a decision tree, holding a predicted value.
    """

    def __init__(self, value, depth=None):
        """
        Initializes a Leaf instance.

        Args:
            value: the predicted value held by this leaf.
            depth (int): the depth of this leaf within the tree.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """
        Returns a human readable string representation of the leaf.

        Returns:
            str: the string representation of the leaf.
        """
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    """
    Represents a decision tree that can be grown, printed, and used to
    predict the target value of individuals based on their explanatory
    features.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initializes a Decision_Tree instance.

        Args:
            max_depth (int): the maximum depth allowed for the tree.
            min_pop (int): the minimum population required to split a
                node further.
            seed (int): the seed used to initialize the random number
                generator.
            split_criterion (str): the criterion used to decide how a
                node is split.
            root (Node): the root node of the tree. If not provided, a
                fresh root Node is created.
        """
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
        """
        Returns a human readable string representation of the whole
        tree.

        Returns:
            str: the string representation of the tree.
        """
        return self.root.__str__()
