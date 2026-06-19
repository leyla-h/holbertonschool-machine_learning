#!/usr/bin/env python3
"""Decision Tree module with training capabilities."""
import numpy as np


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
        self.sub_population = None
        self.lower = {}
        self.upper = {}

    def max_depth_below(self):
        """Return the maximum depth below this node."""
        if self.is_leaf:
            return self.depth
        return max(
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this node."""
        if only_leaves:
            return (
                self.left_child.count_nodes_below(only_leaves=True)
                + self.right_child.count_nodes_below(only_leaves=True)
            )
        return (
            1
            + self.left_child.count_nodes_below()
            + self.right_child.count_nodes_below()
        )

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
                f"root [feature={self.feature},"
                f" threshold={self.threshold}]"
            )
        else:
            node_str = (
                f"-> node [feature={self.feature},"
                f" threshold={self.threshold}]"
            )

        left_str = self.left_child_add_prefix(self.left_child.__str__())
        right_str = self.right_child_add_prefix(self.right_child.__str__())

        return node_str + "\n" + left_str + right_str.rstrip("\n")

    def get_leaves_below(self):
        """Return a list of all leaves below this node."""
        return (
            self.left_child.get_leaves_below()
            + self.right_child.get_leaves_below()
        )

    def update_bounds_below(self):
        """Update the lower and upper bounds for each node below."""
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child, direction in [
            (self.left_child, "left"),
            (self.right_child, "right")
        ]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()
            if direction == "left":
                child.lower[self.feature] = max(
                    child.lower.get(self.feature, -np.inf),
                    self.threshold
                )
            else:
                child.upper[self.feature] = min(
                    child.upper.get(self.feature, np.inf),
                    self.threshold
                )

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Update the indicator function using bounds."""
        def is_large_enough(x):
            return np.all(
                np.array([
                    x[:, key] >= self.lower[key]
                    for key in self.lower
                ]),
                axis=0
            )

        def is_small_enough(x):
            return np.all(
                np.array([
                    x[:, key] < self.upper[key]
                    for key in self.upper
                ]),
                axis=0
            )

        self.indicator = lambda x: (
            np.logical_and(is_large_enough(x), is_small_enough(x))
        )

    def pred(self, x):
        """Return prediction for input x by traversing tree."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Represents a leaf node in a decision tree."""

    def __init__(self, value, depth=0):
        """Initialize a Leaf."""
        super().__init__()
        self.value = value
        self.depth = depth
        self.is_leaf = True
        self.lower = {}
        self.upper = {}
        self.sub_population = None

    def max_depth_below(self):
        """Return the depth of this leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1 since a leaf is a single node."""
        return 1

    def __str__(self):
        """Return string representation of the leaf."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Return this leaf in a list."""
        return [self]

    def update_bounds_below(self):
        """No children to update for a leaf."""
        pass

    def pred(self, x):
        """Return the leaf's value as prediction."""
        return self.value


class Decision_Tree:
    """Represents a decision tree."""

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """Initialize a Decision_Tree."""
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return string representation of the decision tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Return all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds for all nodes."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Update the predict function using leaf indicators."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            result = np.zeros(A.shape[0], dtype=int)
            for leaf in leaves:
                result[leaf.indicator(A)] = leaf.value
            return result

        self.predict = predict

    def pred(self, x):
        """Return prediction for a single sample."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Return the min and max of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Choose a random feature and threshold to split a node."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree on the given data."""
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
            acc = self.accuracy(self.explanatory, self.target)
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {acc}""")

    def fit_node(self, node):
        """Recursively fit a node by splitting into children."""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold
        )

        # Is left node a leaf?
        is_left_leaf = (
            np.sum(left_population) < self.min_pop
            or node.depth + 1 >= self.max_depth
            or np.unique(self.target[left_population]).size == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf?
        is_right_leaf = (
            np.sum(right_population) < self.min_pop
            or node.depth + 1 >= self.max_depth
            or np.unique(self.target[right_population]).size == 1
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Create and return a leaf child node."""
        values, counts = np.unique(
            self.target[sub_population], return_counts=True
        )
        value = values[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create and return a non-leaf child node."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def possible_thresholds(self, node, feature):
        """Return midpoints between consecutive unique feature values."""
        values = np.unique(
            (self.explanatory[:, feature])[node.sub_population]
        )
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Return best threshold and Gini score for one feature."""
        thresholds = self.possible_thresholds(node, feature)
        if thresholds.size == 0:
            return 0, np.inf

        classes = np.unique(self.target[node.sub_population])

        # individuals in sub_population: shape (n,)
        X = self.explanatory[:, feature][node.sub_population]
        y = self.target[node.sub_population]
        n = X.shape[0]

        # goes_left[i, j]: 1 if individual i > threshold j  shape (n, t)
        goes_left = (
            X[:, np.newaxis] > thresholds[np.newaxis, :]
        ).astype(float)
        goes_right = 1.0 - goes_left  # <= threshold

        # is_class[i, k]: 1 if individual i is of class k   shape (n, c)
        is_class = (
            y[:, np.newaxis] == classes[np.newaxis, :]
        ).astype(float)

        # left counts per (threshold, class): shape (t, c)
        left_counts = np.einsum('ij,ik->jk', goes_left, is_class)

        # right counts per (threshold, class): shape (t, c)
        right_counts = np.einsum('ij,ik->jk', goes_right, is_class)

        # total per threshold: shape (t,)
        left_totals = left_counts.sum(axis=1)
        right_totals = right_counts.sum(axis=1)

        # avoid division by zero
        left_totals_safe = np.where(left_totals == 0, 1, left_totals)
        right_totals_safe = np.where(right_totals == 0, 1, right_totals)

        left_probs = left_counts / left_totals_safe[:, np.newaxis]
        right_probs = right_counts / right_totals_safe[:, np.newaxis]

        left_gini = 1 - np.sum(left_probs ** 2, axis=1)
        right_gini = 1 - np.sum(right_probs ** 2, axis=1)

        # weighted average Gini
        gini_avg = (
            left_totals * left_gini + right_totals * right_gini
        ) / n

        best_idx = np.argmin(gini_avg)
        return thresholds[best_idx], gini_avg[best_idx]

    def Gini_split_criterion(self, node):
        """Return the best feature and threshold using Gini impurity."""
        X = np.array([
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])
        ])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]

    def accuracy(self, test_explanatory, test_target):
        """Return the accuracy of predictions on the given data."""
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
