#!/usr/bin/env python3
"""
This module defines the Isolation_Random_Forest class, an ensemble of
Isolation_Random_Tree instances used to score individuals of a
dataset according to how easily they can be isolated, in order to
detect outliers.
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """
    Represents an isolation random forest : an ensemble of isolation
    random trees whose averaged predictions are used to detect the
    outliers of a dataset.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes an Isolation_Random_Forest instance.

        Args:
            n_trees (int): the number of trees to grow in the forest.
            max_depth (int): the maximum depth allowed for each tree.
            min_pop (int): the minimum population required to split a
                node further, for each tree.
            seed (int): the seed used to initialize the random number
                generator of the first tree (tree i uses seed + i).
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts the mean isolation depth of each individual in
        explanatory, averaged over every tree of the forest.

        Args:
            explanatory (numpy.ndarray): the explanatory features of
                the individuals to score.

        Returns:
            numpy.ndarray: the mean depth of each individual, across
                all the trees of the forest.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Builds the forest by growing n_trees isolation random trees
        on explanatory.

        Args:
            explanatory (numpy.ndarray): the explanatory features of
                the training individuals.
            n_trees (int): the number of trees to grow.
            verbose (int): if set to 1, prints a short training
                summary once training is complete.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }""")

    def suspects(self, explanatory, n_suspects):
        """
        Returns the n_suspects rows in explanatory that have the
        smallest mean depth.

        Individuals that are isolated in fewer splits, on average,
        across the forest are the easiest to separate from the rest
        of the population and are therefore considered the most
        likely outliers.

        Args:
            explanatory (numpy.ndarray): the explanatory features of
                the individuals to inspect.
            n_suspects (int): the number of suspects to return.

        Returns:
            tuple: (suspects, depths) where suspects is the
                numpy.ndarray containing the n_suspects rows of
                explanatory with the smallest mean depth, and depths
                is the numpy.ndarray of their corresponding mean
                depths, both ordered from the smallest depth to the
                largest.
        """
        depths = self.predict(explanatory)
        indices = np.argsort(depths)[:n_suspects]
        return explanatory[indices], depths[indices]
