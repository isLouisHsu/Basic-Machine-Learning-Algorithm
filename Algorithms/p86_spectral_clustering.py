import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder

# from p78_kmeans import KMeans

def loadData():
    n_clusters = 2
    
    x = np.arange(-5, 5, 0.02)
    y = np.sqrt(25 - x**2) 
    x = x + np.random.rand(x.shape[0]) - 0.5
    y = y + np.random.rand(y.shape[0]) - 0.5 
    X1 = np.c_[x, y]
    n1 = X1.shape[0]
    y1 = np.zeros(n1)
    
    mu = np.array([0, 2])
    s = np.array([[1, 0], [0, 1]])
    n2 = 100
    X2 = np.dot(np.random.randn(n2, 2), np.linalg.cholesky(s)) + mu
    y2 = np.ones(n2)
    
    X = np.r_[X1, X2]
    y = np.r_[y1, y2]

    return X, y

def showfig(X, y):
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

def score(y_true, y_pred):
    return np.mean(y_true==y_pred.astype('float'))

class SpectralClustering():
    """
    Attributes:
        k: {int}, k < n_samples
        sigma: {float}
    Notes:
        Steps:
            - similarity matrix [W_{n×n}]
            - diagonal matrix [D_{n×n}] is defined as
                    D_{ii} = \begin{cases}
                                \sum_j W_{ij} & i \neq j \\
                                0 & i = j
                            \end{cases}
            - Laplacian matrix [L_{n×n}], Laplacian matrix is defined as
                    L = D - W   or	L = D^{-1} (D - W)
            - EVD: L \alpha_i = \lambda_i \alpha_i
            - takes the eigenvector corresponding to the largest eigenvalue as
                    B_{n×k} = [\beta_1, \beta_2, ..., \beta_k]
            - apply K-Means to the row vectors of matrix B
    """
    def __init__(self, k, n_clusters=2, sigma=1.0):
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.k = k
        self.sigma = sigma
    def predict(self, X):
        n_samples = X.shape[0]
        # step 1
        kernelGaussian = lambda z, sigma: np.exp(-0.5 * np.square(z/sigma))
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i):
                W[i, j] = kernelGaussian(np.linalg.norm(X[i] - X[j]), self.sigma)
                W[j, i] = W[i, j]
        # step 2
        D = np.diag(np.sum(W, axis=1))
        # step 3
        L = D - W
        L = np.linalg.inv(D).dot(L)
        # step 4
        eigval, eigvec = np.linalg.eig(L)
        # step 5
        order = np.argsort(eigval)
        eigvec = eigvec[:, order]
        beta = eigvec[:, :self.k]
        # step 6
        self.kmeans.fit(beta)
        return self.kmeans.labels_

if __name__ == '__main__':
    X, y = loadData()
    # showfig(X, y)

    y_pred = SpectralClustering(k=3, n_clusters=2, sigma=0.3).predict(X)
    showfig(X, y_pred)