#!/usr/bin/env python3
"""This module defines an Isolation Random Tree class."""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """Represents an isolation random tree."""
    def __init__(self, max_depth=10, seed=0, root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def depth(self):
        def get_depth(node):
            if node.is_leaf:
                return node.depth
            return max(get_depth(node.left_child), get_depth(node.right_child))
        return get_depth(self.root)

    def count_nodes(self, only_leaves=False):
        def count(node):
            if node.is_leaf:
                return 1
            res = count(node.left_child) + count(node.right_child)
            return res if only_leaves else res + 1
        return count(self.root)

    def update_bounds(self):
        self.root.update_bounds_below()

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_predict(self):
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([leaf.value for leaf in leaves])[
            np.argmax([leaf.indicator(A) for leaf in leaves], axis=0)
        ]

    def random_split_criterion(self, node):
        feat = self.rng.integers(0, self.explanatory.shape[1])
        vals = self.explanatory[node.sub_population, feat]
        return feat, self.rng.uniform(np.min(vals), np.max(vals))

    def get_leaf_child(self, node, sub_population):
        # In isolation trees, the value is the depth of the leaf
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        node.feature, node.threshold = self.random_split_criterion(node)
        feat_vals = self.explanatory[:, node.feature]
        
        left_population = (feat_vals > node.threshold) & node.sub_population
        right_population = (feat_vals <= node.threshold) & node.sub_population

        # Leaf condition: reach max_depth or population too small to split
        is_left_leaf = (node.depth + 1 == self.max_depth) or (np.sum(left_population) <= self.min_pop)
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth + 1 == self.max_depth) or (np.sum(right_population) <= self.min_pop)
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print(f"  Training finished.\n"
                  f"    - Depth                     : {self.depth()}\n"
                  f"    - Number of nodes           : {self.count_nodes()}\n"
                  f"    - Number of leaves          : {self.count_nodes(only_leaves=True)}")
