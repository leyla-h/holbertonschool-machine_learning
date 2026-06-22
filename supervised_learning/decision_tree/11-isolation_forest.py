#!/usr/bin/env python3
"""Module implementing the Isolation Random Forest for anomaly detection."""
import numpy as np

Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """Isolation Random Forest for detecting outliers/anomalies.

    This class builds an ensemble of Isolation Random Trees and uses
    the mean depth of each data point across all trees to identify
    suspects (anomalies): points with the smallest mean depth are
    isolated most easily, indicating they are outliers.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize the Isolation Random Forest.

        Parameters
        ----------
        n_trees : int
            Number of trees in the forest.
        max_depth : int
            Maximum depth allowed for each tree.
        min_pop : int
            Minimum population required to split a node.
        seed : int
            Random seed for reproducibility.
        """
        # FIX 1: removed the stale `self.numpy_predicts = []` attribute that
        #         was never used anywhere (the real list is self.numpy_preds).
        self.numpy_preds = None
        self.target = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        # FIX 2: store min_pop as a forest attribute (used internally only).
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predict the mean depth for each individual in explanatory.

        Parameters
        ----------
        explanatory : np.ndarray
            Array of shape (n_samples, n_features) with input data.

        Returns
        -------
        np.ndarray
            Mean depth across all trees for each sample.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, verbose=0):
        """Fit the Isolation Random Forest on the given data.

        Parameters
        ----------
        explanatory : np.ndarray
            Array of shape (n_samples, n_features) with training data.
        verbose : int
            If 1, prints training summary statistics.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(self.n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth,
                seed=self.seed + i,
                # min_pop belongs to the forest only; Isolation_Random_Tree
                # does not expose that parameter.
            )
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """Return the n_suspects rows with the smallest mean depth.

        Points that are isolated with the fewest splits (smallest mean
        depth) are considered the most anomalous/suspicious.

        Parameters
        ----------
        explanatory : np.ndarray
            Array of shape (n_samples, n_features) with input data.
        n_suspects : int
            Number of suspects (anomalies) to return.

        Returns
        -------
        tuple
            A tuple (suspects, depths) where suspects is an array of
            shape (n_suspects, n_features) and depths is an array of
            the corresponding mean depths.
        """
        depths = self.predict(explanatory)
        indices = np.argsort(depths)[:n_suspects]
        return explanatory[indices], depths[indices]
