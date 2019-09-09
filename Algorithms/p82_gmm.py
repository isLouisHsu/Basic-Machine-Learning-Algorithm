import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_data():
    mu1 = np.array([[6, 6]])
    s1 = np.array([[1, -0.5], [-0.5, 1]])
    X1 = np.dot(np.random.randn(150, 2), nl.cholesky(s1)) + mu1
    mu2 = np.array([[10, 10]])
    s2 = np.array([[2, 0.5], [0.5, 1]])
    X2 = np.dot(np.random.randn(50, 2), nl.cholesky(s2)) + mu2
    mu3 = np.array([[10, 0]])
    s3 = np.array([[2, 0.5], [0.5, 2]])
    X3 = np.dot(np.random.randn(100, 2), nl.cholesky(s3)) + mu3
    X = np.r_[X1, X2, X3]
    return X
def multiGaussian(x, mu, sigma):
    n = x.shape[0]
    z = (x-mu).T.dot(nl.inv(sigma)).dot(x-mu)
    return np.exp(-0.5*z)/np.sqrt(2*np.pi*nl.det(sigma))


class GMM():
    """ Gaussian Mixture Model
    Attributes:
        n_clusters {int}
        prior {ndarray(n_clusters,)}
        mu {ndarray(n_clusters, n_features)}
        sigma {ndarray(n_clusters, n_features, n_features)}
    """
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.prior = None
        self.mu = None
        self.sigma = None
    def fit(self, X, delta=0.01):
        """
        Args:
            X {ndarray(n_samples, n_features)}
            delta {float}
        Notes:
            - Initialize with k-means
        """
        (n_samples, n_features) = X.shape

        # initialize with k-means
        clf = KMeans(n_clusters=self.n_clusters)
        clf.fit(X)
        self.mu = clf.cluster_centers_ 
        self.prior = np.zeros(self.n_clusters)
        self.sigma = np.zeros((self.n_clusters, n_features, n_features))
        for k in range(self.n_clusters):
            X_ = X[clf.labels_==k]
            self.prior[k] = X_.shape[0] / X_.shape[0]
            self.sigma[k] = np.cov(X_.T)
        
        while True:
            mu_ = self.mu.copy()
            # E-step: updata gamma
            gamma = np.zeros((n_samples, self.n_clusters))
            for i in range(n_samples):
                for k in range(self.n_clusters):
                    denominator = 0
                    for j in range(self.n_clusters):
                        post = self.prior[k] *\
                                    multiGaussian(X[i], self.mu[j], self.sigma[j])
                        denominator += post
                        if j==k: numerator = post
                    gamma[i, k] = numerator/denominator
            # M-step: updata prior, mu, sigma
            for k in range(self.n_clusters):
                sum1 = 0
                sum2 = 0
                sum3 = 0
                for i in range(n_samples):
                    sum1 += gamma[i, k]
                    sum2 += gamma[i, k] * X[i]
                    x_ = np.reshape(X[i] - self.mu[k], (n_features, 1))
                    sum3 += gamma[i, k] * x_.dot(x_.T)
                self.prior[k]  = sum1 / n_samples
                self.mu[k]     = sum2 / sum1
                self.sigma[k]  = sum3 / sum1
            # to stop
            mu_delta = 0
            for k in range(self.n_clusters):
                mu_delta += nl.norm(self.mu[k] - mu_[k])
            print(mu_delta)
            if mu_delta < delta: break
        return self.prior, self.mu, self.sigma
    def predict_proba(self, X):
        """
        Args:
            X {ndarray(n_samples, n_features)}
        Returns:
            y_pred_proba {ndarray(n_samples, n_clusters)}
        """
        (n_samples, n_features) = X.shape
        y_pred_proba = np.zeros((n_samples, self.n_clusters))
        for i in range(n_samples):
            for k in range(self.n_clusters):
                y_pred_proba[i, k] = self.prior[k] *\
                                multiGaussian(X[i], self.mu[k], self.sigma[k])
        return y_pred_proba
    def predict(self, X):
        """
        Args:
            X {ndarray(n_samples, n_features)}
        Returns:
            y_pred_proba {ndarray(n_samples,)}
        """
        y_pred_proba = self.predict_proba(X)
        return np.argmax(y_pred_proba, axis=1)

if __name__ == '__main__':
    X = load_data()

    estimator_kmeans = KMeans(n_clusters=3)
    estimator_kmeans.fit(X)
    y_pred_kmeans = estimator_kmeans.predict(X)

    estimator_gmm = GMM(n_clusters=3)
    estimator_gmm.fit(X, delta=0.01)
    y_pred_gmm = estimator_gmm.predict(X)

    fx = np.arange(300)
    plt.figure(); plt.axvline(150); plt.axvline(200)
    plt.title("GMM")
    plt.scatter(fx, y_pred_gmm)
    plt.figure(); plt.axvline(150); plt.axvline(200)
    plt.title("KMeans")
    plt.scatter(fx, y_pred_kmeans)
    plt.show()
