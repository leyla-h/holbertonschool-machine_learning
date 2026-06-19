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
            self.upper = {i: np.inf for i in range(100)}
            self.lower = {i: -np.inf for i in range(100)}
        for child in [self.left_child, self.right_child]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()
        self.left_child.upper[self.feature] = self.threshold
        self.right_child.lower[self.feature] = self.threshold
        self.left_child.update_bounds_below()
        self.right_child.update_bounds_below()

    def update_indicator(self):
        self.left_child.update_indicator()
        self.right_child.update_indicator()


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


class Decision_Tree:
    """Represents a decision tree."""
    def __init__(self, root=None, split_criterion="Gini", max_depth=10, min_pop=1, seed=0):
        self.root = root if root else Node(is_root=True)
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.rng = np.random.default_rng(seed)

    def fit(self, explanatory, target, verbose=0):
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
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
            is_leaf = (len(vals) <= self.min_pop) or (len(np.unique(vals)) <= 1)
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
        f = self.rng.integers(0, self.explanatory.shape[1])
        vals = self.explanatory[node.sub_population, f]
        return f, self.rng.uniform(np.min(vals), np.max(vals))

    def possible_thresholds(self, node, feature):
        vals = np.unique(self.explanatory[node.sub_population, feature])
        return (vals[1:] + vals[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        thresholds = self.possible_thresholds(node, feature)
        if thresholds.size == 0: return np.inf, None
        targets = self.target[node.sub_population]
        feat_vals = self.explanatory[node.sub_population, feature]
        
        # Vectorized Gini Calculation
        left_mask = feat_vals[:, None] > thresholds
        n_left = left_mask.sum(axis=0)
        n_right = len(targets) - n_left
        
        gini = np.ones_like(thresholds)
        for cls in np.unique(self.target):
            cls_mask = (targets == cls)[:, None]
            p_left = (left_mask & cls_mask).sum(axis=0) / (n_left + 1e-9)
            p_right = ((~left_mask) & cls_mask).sum(axis=0) / (n_right + 1e-9)
            gini -= (p_left**2 + p_right**2)
            
        best = np.argmin(gini)
        return gini[best], thresholds[best]

    def Gini_split_criterion(self, node):
        res = [self.Gini_split_criterion_one_feature(node, i) for i in range(self.explanatory.shape[1])]
        best_f = np.argmin([r[0] for r in res])
        return best_f, res[best_f][1]

    def update_predict(self):
        self.update_bounds(); leaves = self.get_leaves()
        for leaf in leaves: leaf.update_indicator()
        self.predict = lambda A: np.array([leaf.value for leaf in leaves])[np.argmax([leaf.indicator(A) for leaf in leaves], axis=0)]

    def depth(self): return max([l.depth for l in self.get_leaves()])
    def count_nodes(self, only_leaves=False):
        def count(node):
            if node.is_leaf: return 1 if only_leaves else 1
            return (1 if not only_leaves else 0) + count(node.left_child) + count(node.right_child)
        return count(self.root)
    def accuracy(self, te, tt): return np.sum(self.predict(te) == tt) / tt.size
    def get_leaves(self): return self.root.get_leaves_below()
    def update_bounds(self): self.root.update_bounds_below()
