import numpy as np
import pandas as pd


class K_means:
    def __init__(self, k, data, distance):
        self.k = k # number of clusters
        self.data = data # data to be clustered
        self.distance = distance # distance metric type
        self.centroids = [] # centroids of clusters
        self.clusters = []  # cluster of each data point
    
    
    def update_centroids(self, x):
        return np.mean(x, axis=0)

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
    
    def minkowski_distance(self, x, y):
        sum = 0
        for i in range(x.shape[0]):
            sum += abs(x[i] - y[i])**3
        return sum**(1/3)   


    def closest_cluster(self, instances, instance, distance):
        if distance == 'euclidean':
            distances = [self.euclidean_distance(instance, inst) for inst in instances]
            return np.argmin(distances)
        if distance == 'manhattan':
            distances = [self.manhattan_distance(instance, inst) for inst in instances]
            return np.argmin(distances)
        if distance == 'cosine':
            distances = [self.cosine_distance(instance, inst) for inst in instances]
            return np.argmin(distances)
        if distance == 'hamming':
            distances = [self.hamming_distance(instance, inst) for inst in instances]
            return np.argmin(distances)
        if distance == 'minkowski':
            distances = [self.minkowski_distance(instance, inst) for inst in instances]
            return np.argmin(distances)
        
        
    def fit(self,max_iters=100):
            self.centroids = self.data[np.random.choice(len(self.data), self.k, replace=False)]

            for _ in range(max_iters):
                self.labels = np.array([self.closest_cluster(self.centroids, instance, self.distance) for instance in self.data])

                new_centroids = np.array([np.mean(self.data[self.labels == i], axis=0) for i in range(self.k)])

                if np.all(np.isclose(self.centroids, new_centroids, rtol=1e-4)): 
                    break

                self.centroids = new_centroids

            return self.labels, self.centroids 
        
    



            