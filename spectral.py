import numpy as np
from sklearn.cluster import KMeans

class Spectral:
    def __init__(self, n_clusters, gamma=1):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)


    def fit(self, X):
        n_smp = X.shape[0]
        W = np.zeros((n_smp, n_smp))
        D = np.zeros((n_smp, n_smp))

        for i in range(n_smp):
            for j in range(n_smp):
                W[i][j] = np.exp(-self.gamma*np.linalg.norm(X[i]-X[j])**2)

        D = np.diag(np.sum(W, axis=1))

        D_inv = np.sqrt(np.linalg.inv(D))
        L = np.eye(n_smp)-D_inv @ W @ D_inv

        e_values, self.e_vectors = np.linalg.eigh(L)
        self.e_vectors = self.e_vectors[:, :self.n_clusters]


    def predict(self):
        self.kmeans.fit(self.e_vectors)
        labels = self.kmeans.labels_
        return labels


