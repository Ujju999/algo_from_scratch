import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class KMeans:
    def __init__(self,n_clusters = 5, max_iterations = 500, random_state = 53,convergence_threshold = 1e-6):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        np.random.seed(random_state)
        self.convergence_threshold = convergence_threshold

        self.centroids = None
        self.labels = None
        self.inertia = None

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        n_samples,n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters,replace = False)

        self.centroids = X[random_indices]
        for iteration in range(self.max_iterations):
            distances = np.zeros((n_samples, self.n_clusters))
            for j in range(self.n_clusters):
                distances[:,j] = np.linalg.norm(X - self.centroids[j],axis = 1)

            self.labels = np.argmin(distances, axis = 1)

            old_centroids = self.centroids.copy()
            for k in range(self.n_clusters):
                cluster_points = X[self.labels == k]
                if len(cluster_points) > 0:
                    self.centroids[k] = np.mean(cluster_points, axis = 0)
                else:
                    self.centroids[k] = X[np.random.choice(n_samples)]

            centroid_shift = np.linalg.norm(self.centroids - old_centroids)

            if centroid_shift < self.convergence_threshold:
                print(f"Converged at iteration {iteration}")
                break

        final_centroids = self.centroids[self.labels]
        distances = np.linalg.norm(X - final_centroids,axis = 1)
        self.inertia = np.sum(distances ** 2)

        return self


    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))

        for j in range(self.n_clusters):
            distances[:,j] = np.linalg.norm(X - self.centroids[j], axis = 1)

        return np.argmin(distances, axis = 1)


