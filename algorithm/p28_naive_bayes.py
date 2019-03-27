import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def multigaussian(x, mu, sigma):
    """ 多维高斯分布
    Args:
        x: {ndarray(n_features,)}
        mu: {ndarray(n_features,)}
        sigma: {ndarray(n_features, n_features)}
    Notes:
        f(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp (-\frac{1}{2} (x-mu)^T \Sigma^{-1} (x-mu))
    """
    x -= mu; n = x.shape[0]
    return np.exp(- 0.5 * x.T.dot(np.linalg.inv(sigma)).dot(x)) / np.sqrt((2*np.pi)**n * np.linalg.det(sigma))

def gaussian(x, mu, sigma):
    return np.exp(-0.5*np.square((x-mu)/sigma))/(np.sqrt(2*np.pi)*sigma)

class NaiveBayes():
    """
    Attributes:
        prior: {ndarray(n_classes,)} priori probability
        mu: {ndarray(n_classes, n_features)}
        sigma: {ndarray(n_classes, n_features)}
    Notes:
        假定
            - 每个维度均服从高斯分布
            - 各个维度间独立，协方差为0
    """
    def __init__(self):
        self.prior = None
        self.mu = None
        self.sigma = None
        self.n_classes = None
        self.n_features = None
    def fit(self, X, y):
        """ 训练模型
        Args:
            X: {ndarray(n_samples, n_features)}
            y: {ndarray(n_samples,)}
        """
        labels = list(set(y))
        self.n_classes = len(labels)
        self.n_features = X.shape[1]

        n_samples = y.shape[0]
        self.prior = np.bincount(y) / n_samples
        self.mu = np.zeros(shape=(self.n_classes, self.n_features))
        self.sigma = np.zeros(shape=(self.n_classes, self.n_features))
        for c in range(self.n_classes):
            X_ = X[y==labels[c]]
            self.mu[c] = np.mean(X_, axis=0)
            self.sigma[c] = np.std(X_, axis=0)
    def predict(self, X):
        """ predict
        Args:
            X: {ndarray(n_samples, n_features)}
        Notes:
            if g(c_i) > g(c_j) then x \in c_i
            g(c_i) is defined as:
                    g(c_i) = p(x|c_i)p(c_i)
        """
        n_samples = X.shape[0]
        y = np.zeros(shape=(n_samples, self.n_classes))
        
        for i in range(n_samples):
            for c in range(self.n_classes):
                y[i, c] = self.prior[c]
                for j in range(self.n_features):
                    y[i, c] *= gaussian(X[i, j], self.mu[c, j], self.sigma[c, j])

        y = y / np.sum(y, axis=1).reshape(-1, 1)
        return np.argmax(y, axis=1)
    def score(self, y_true, y_pred):
        """ accuracy score
        """
        return np.mean(y_true==y_pred)
    def showfig(self, X, y):
        y_pred = self.predict(X)
        n_samples = np.arange(0, X.shape[0])
        plt.figure()

        plt.subplot(221)
        plt.title('feature1')
        plt.axvline(50);plt.axvline(100)
        plt.scatter(n_samples, X[:, 0], c=y_pred)

        plt.subplot(222)
        plt.title('feature2')
        plt.axvline(50);plt.axvline(100)
        plt.scatter(n_samples, X[:, 1], c=y_pred)

        plt.subplot(223)
        plt.title('feature3')
        plt.axvline(50);plt.axvline(100)
        plt.scatter(n_samples, X[:, 2], c=y_pred)

        plt.subplot(224)
        plt.title('feature4')
        plt.axvline(50);plt.axvline(100)
        plt.scatter(n_samples, X[:, 3], c=y_pred)
        plt.show()

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)

    estimator = NaiveBayes()
    estimator.fit(X, y)

    y_pred = estimator.predict(X)

    acc = estimator.score(y, y_pred)
    print("accuracy score is {:>3.2f}%".format(acc*100))

    estimator.showfig(X, y)