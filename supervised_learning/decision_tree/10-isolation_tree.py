#!/usr/bin/env python3
"""
Defines the Isolation_Random_Tree class for outlier detection
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    Represents an isolation random tree for outlier detection
    """
    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initializes an Isolation_Random_Tree instance
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        Returns string representation of the tree
        """
        return self.root.__str__().rstrip('\n')

    def depth(self):
        """
        Returns the maximum depth of the tree
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Returns the total number of nodes or leaves in the tree
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """
        Computes bounds for all nodes in the tree
        """
        self.root.update_bounds_below()

    def get_leaves(self):
        """
        Returns all leaves of the tree
        """
        return self._get_leaves_below(self.root)

    def _get_leaves_below(self, node):
        """
        Recursively collects all leaves below a node
        """
        if node.is_leaf:
            return [node]
        leaves = []
        if node.left_child:
            leaves += self._get_leaves_below(node.left_child)
        if node.right_child:
            leaves += self._get_leaves_below(node.right_child)
        return leaves

    def update_predict(self):
        """
        Updates the predict function using leaf depth as prediction value
        """
        self.predict = lambda A: np.array(
            [self.root.pred(x) for x in A])

    def np_extrema(self, arr):
        """
        Returns the min and max of an array
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Randomly selects a feature and threshold to split a node
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Creates and returns a leaf child whose value is its depth
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates and returns an internal child node
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Recursively fits a node by splitting or making it a leaf
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = (
            node.sub_population &
            (self.explanatory[:, node.feature] > node.threshold)
        )
        right_population = (
            node.sub_population &
            (self.explanatory[:, node.feature] <= node.threshold)
        )

        is_left_leaf = (
            np.sum(left_population) <= self.min_pop or
            node.depth + 1 >= self.max_depth
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            np.sum(right_population) <= self.min_pop or
            node.depth + 1 >= self.max_depth
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Trains the isolation tree on the given data
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(
            explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
