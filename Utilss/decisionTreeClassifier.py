import collections
import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin


class MyDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=float('inf'), min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, dataset, targets):
        assert targets.nunique() >= 2
        targets = targets.to_frame(name='label')
        self.tree = self._build_single_tree(dataset, targets, depth=0)

    def predict(self, dataset):
        res = []
        for _, row in dataset.iterrows():
            pred_value = self._predict_single_row(row)
            res.append(pred_value)
        return np.array(res)

    def _predict_single_row(self, row):
        node = self.tree
        while node.leaf_value is None:
            if row[node.split_feature] <= node.split_value:
                node = node.tree_left
            else:
                node = node.tree_right
        return node.leaf_value

    def _build_single_tree(self, dataset, targets, depth):
        if len(targets['label'].unique()) == 1 or dataset.__len__() <= self.min_samples_split or depth == self.max_depth:
            tree = Tree()
            tree.leaf_value = self._calc_leaf_value(targets['label'])
            return tree

        best_split_feature, best_split_value = self._choose_best_feature(dataset, targets)
        left_dataset, right_dataset, left_targets, right_targets = \
            self._split_dataset(dataset, targets, best_split_feature, best_split_value)

        tree = Tree()
        if left_dataset.__len__() <= self.min_samples_leaf or right_dataset.__len__() <= self.min_samples_leaf:
            tree.leaf_value = self._calc_leaf_value(targets['label'])
            return tree

        tree.split_feature = best_split_feature
        tree.split_value = best_split_value
        tree.tree_left = self._build_single_tree(left_dataset, left_targets, depth + 1)
        tree.tree_right = self._build_single_tree(right_dataset, right_targets, depth + 1)
        return tree

    def _choose_best_feature(self, dataset, targets):
        best_split_feature = None
        best_split_value = None
        best_gini = 1

        for feature in dataset.columns:
            unique_values = sorted(dataset[feature].unique().tolist())
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                gini = self._calc_gini(left_targets['label'], right_targets['label'])

                if gini < best_gini:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_gini = gini

        return best_split_feature, best_split_value

    @staticmethod
    def _calc_leaf_value(targets):
        label_counts = collections.Counter(targets)
        major_label = max(label_counts, key=label_counts.get)
        return major_label

    @staticmethod
    def _calc_gini(left_targets, right_targets):
        gini = 0
        for targets in [left_targets, right_targets]:
            label_counts = collections.Counter(targets)
            total_samples = len(targets)
            impurity = 1.0

            for key in label_counts:
                prob = label_counts[key] / total_samples
                impurity -= prob ** 2

            gini += total_samples * impurity

        total_samples = len(left_targets) + len(right_targets)
        weighted_gini = gini / total_samples
        return weighted_gini

    @staticmethod
    def _split_dataset(dataset, targets, split_feature, split_value):
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets


class Tree(object):
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None
