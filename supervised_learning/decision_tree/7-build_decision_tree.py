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

    def pred(self, x):
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)

    def get_leaves_below(self):
        return self.left_child.get_leaves_below() + self.right_child.get_leaves_below()

    def update_bounds_below(self):
        if self.is_root:
            self.upper = {i: np.inf for i in range(100)}  # Adjust size if needed
            self.lower = {i: -np.inf for i in range(100)}
        for child in [self.left_child, self.right_child]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

        # left  = "> threshold"  -> lower bound is threshold
        # right = "<= threshold" -> upper bound is threshold
        self.left_child.lower[self.feature] = self.threshold
        self.right_child.upper[self.feature] = self.threshold

        self.left_child.update_bounds_below()
        self.right_child.update_bounds_below()

    def update_indicator(self):
        self.left_child.update_indicator()
        self.right_child.update_indicator()

    def max_depth_below(self):
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        return (1 + self.left_child.count_nodes_below(only_leaves=only_leaves) +
                self.right_child.count_nodes_below(only_leaves=only_leaves))


class Leaf:
    """Represents a leaf in a decision tree."""
    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.is_leaf = True
        self.sub_population = None

    def pred(self, x):
        return self.value

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        pass

    def update_indicator(self):
        def is_large_enough(x):
            res = [np.greater(x[:, key], val) for key, val in self.lower.items() if val != -np.inf]
            return np.all(res, axis=0) if res else np.full(x.shape[0], True)

        def is_small_enough(x):
            res = [np.less_equal(x[:, key], val) for key, val in self.upper.items() if val != np.inf]
            return np.all(res, axis=0) if res else np.full(x.shape[0], True)
        self.indicator = lambda x: np.all([is_large_enough(x), is_small_enough(x)], axis=0)

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1


class Decision_Tree:
    """Represents a decision tree."""
    def __init__(self, root=None, split_criterion="random", max_depth=10, min_pop=1, seed=0):
        self.root = root if root else Node(is_root=True)
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.rng = np.random.default_rng(seed)

    def fit(self, explanatory, target, verbose=0):
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print(f"  Training finished.\n- Depth                     : {self.depth()}\n- Number of nodes           : {self.count_nodes()}\n- Number of leaves          : {self.count_nodes(only_leaves=True)}\n- Accuracy on training data : {self.accuracy(self.explanatory, self.target)}")

    def fit_node(self, node):
        node.feature, node.threshold = self.split_criterion(node)
        left_mask = (self.explanatory[:, node.feature] > node.threshold) & node.sub_population
        right_mask = (self.explanatory[:, node.feature] <= node.threshold) & node.sub_population

        for mask, side in [(left_mask, 'left'), (right_mask, 'right')]:
            vals = self.target[mask]
            is_leaf = (len(vals) <= self.min_pop) or (len(np.unique(vals)) <= 1) or (node.depth + 1 >= self.max_depth)
            child = self.get_leaf_child(node, mask) if is_leaf else self.get_node_child(node, mask)
            if side == 'left': node.left_child = child
            else: node.right_child = child
            if not is_leaf: self.fit_node(child)

    def get_leaf_child(self, node, sub_pop):
        vals, counts = np.unique(self.target[sub_pop], return_counts=True)
        leaf = Leaf(vals[np.argmax(counts)])
        leaf.depth, leaf.sub_population = node.depth + 1, sub_pop
        return leaf

    def get_node_child(self, node, sub_pop):
        n = Node(); n.depth, n.sub_population = node.depth + 1, sub_pop
        return n

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            f = self.rng.integers(0, self.explanatory.shape[1])
            f_min, f_max = np.min(self.explanatory[node.sub_population, f]), np.max(self.explanatory[node.sub_population, f])
            diff = f_max - f_min
        return f, self.rng.uniform(f_min, f_max)

    def update_predict(self):
        self.update_bounds(); leaves = self.get_leaves()
        for leaf in leaves: leaf.update_indicator()
        self.predict = lambda A: np.array([leaf.value for leaf in leaves])[np.argmax([leaf.indicator(A) for leaf in leaves], axis=0)]

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def accuracy(self, te, tt): return np.sum(self.predict(te) == tt) / tt.size
    def get_leaves(self): return self.root.get_leaves_below()
    def update_bounds(self): self.root.update_bounds_below()
