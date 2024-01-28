import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(instance) for instance in X_test.values]
        return np.array(predictions)

    def _predict(self, instance):
        sorted_instances = self._sort_instances(instance)
        k_nearest_neighbors = [self.y_train[index] for index, _ in sorted_instances[:self.k]]
        return self._dominant_class(k_nearest_neighbors)

    def _sort_instances(self, instance):
        distances = self.X_train.apply(lambda row: self._euclidean_distance(instance, row), axis=1)
        sorted_instances = distances.sort_values().reset_index()
        return sorted_instances[['index', 0]].values.tolist()

    def _euclidean_distance(self, inst1, inst2):
        return np.sqrt(np.sum((np.array(inst1) - np.array(inst2))**2))

    def _dominant_class(self, labels):
        counter = Counter(labels)
        most_common = counter.most_common(1)
        return most_common[0][0]
    
