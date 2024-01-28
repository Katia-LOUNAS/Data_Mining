import numpy as np
import matplotlib.pyplot as plt


class DBSCAN:
    def __init__(self, eps, min_samples, distance):
        self.eps = eps
        self.min_samples = min_samples
        self.distance = distance
        self.noise = []
        self.core_samples = []
        self.neighbors = []
        self.neighborhood = []
        self.neighborhoods = []
        self.distance_matrix = []
        self.labels = []
        self.clusters = []
        
        
    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))
    
    def manhattan_distance(self, x, y):     
        sum = 0
        for i in range(x.shape[0]):
            sum += abs(x[i] - y[i])
        return sum
    
    def cosine_distance(self, x, y):
        sum = 0
        sum1 = 0
        sum2 = 0
        for i in range(x.shape[0]):
            sum += x[i] * y[i]
            sum1 += x[i] * x[i]
            sum2 += y[i] * y[i]
            
        return 1 - (sum / sum1 * sum2)
    
    def hamming_distance(self, x, y):
        return sum(x[i] != y[i] for i in range(len(x)))
    
    def eps_neighborhood(self, x, y, eps, distance):
        if distance == 'euclidean':
            return self.euclidean_distance(x, y) < eps
        if distance == 'manhattan':
            return self.manhattan_distance(x, y) < eps
        if distance == 'cosine':
            return self.cosine_distance(x, y) < eps
        if distance == 'hamming':
            return self.hamming_distance(x, y) < eps
        
    
    # get
    def region_query(self, point, eps, distance):
        neighbors = []
        for i in range(len(self.X)):
            if self.eps_neighborhood(point, self.X[i], eps, distance):
                neighbors.append(i)
        return neighbors
    
    def expand_cluster(self, point, neighbors, cluster, eps, min_samples, distance):
        self.labels[point] = cluster
        i = 0
        while i < len(neighbors):
            p = neighbors[i]
            if self.labels[p] == -1:
                self.labels[p] = cluster
            elif self.labels[p] == 0:
                self.labels[p] = cluster
                p_neighbors = self.region_query(self.X[p], eps, distance)
                if len(p_neighbors) >= min_samples:
                    neighbors = neighbors + p_neighbors
            i += 1
        
    def fit(self, X):
        self.X = X
        self.labels = [0] * len(self.X)
    
        self.distance_matrix = [[self.euclidean_distance(self.X[i], self.X[j]) for j in range(len(self.X))] for i in range(len(self.X))]
        for i in range(len(self.X)):
            self.neighbors.append(self.region_query(self.X[i], self.eps, self.distance))
            if len(self.neighbors[i]) >= self.min_samples:
                self.core_samples.append(i)
                self.neighborhood.append(self.neighbors[i])
                self.neighborhoods.append(self.neighbors[i])
        self.clusters = []
        cluster = 1
        for i in range(len(self.X)):
            if self.labels[i] == 0:
                if i in self.core_samples:
                    self.clusters.append(cluster)
                    self.expand_cluster(i, self.neighbors[i], cluster, self.eps, self.min_samples, self.distance)
                    cluster += 1
                else:
                    self.labels[i] = -1
                    self.noise.append(i)
        return self.labels, self.clusters, self.noise, self.core_samples, self.neighbors, self.neighborhood, self.neighborhoods, self.distance_matrix
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels
    
    def plot(self, X):
        plt.scatter(X[:, 0], X[:, 1], c=self.labels, cmap='viridis', edgecolors='k', alpha=0.7)
        plt.title('DBSCAN Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()