from DecisionTree import DecisionTree
import pandas as pd
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    def _train_single_tree(self, idxs, X, y):
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.n_features,
        )
        X_sample, y_sample = X.iloc[idxs], y.iloc[idxs]
        tree.fit(X_sample, y_sample)
        return tree

    def fit(self, X, y):
        n_samples = X.shape[0]
        idxs_list = [np.random.choice(n_samples, n_samples, replace=True) for _ in range(self.n_trees)]
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._train_single_tree, idxs, X, y)
                for idxs in idxs_list
            ]
        self.trees = [future.result() for future in futures]

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        # return X[idxs], y[idxs]
        return X.iloc[idxs], y.iloc[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
