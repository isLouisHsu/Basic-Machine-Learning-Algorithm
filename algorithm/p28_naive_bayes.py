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

import numpy as np
import matplotlib.pyplot as plt

class NaiveBayesLog(object):
    """ Naive Bayes estimator
                   p(x|w_i)p(w_i)
    p(w_i|x) = ----------------------
                \sum_j p(x|w_j)p(w_j)
    Attributes:
        means_:         {ndarray(n_class, n_feats)}
        covariances_:   {ndarray(n_class, n_feats, n_feats)}
    """
    def __init__(self):
        
        self.prior_ = None
        self.means_ = None
        self.covariances_ = None
    
    def fit(self, X, y):
        """ train estimator
        
        Params:
            X: {ndarray(n_samples, n_feats)}
            y: {ndarray(n_samples)}          0, 1, 2, ...
        """
        n_class = len(set(list(y)))
        n_feats = X.shape[1]

        ## initialize
        self.prior_ = np.zeros(shape=n_class)
        self.means_ = np.zeros(shape=(n_class, n_feats))
        self.covariances_ = np.zeros(shape=(n_class, n_feats, n_feats))

        ## calculate mean and covariance of each class
        for i_class in range(n_class):
            idx = y==i_class
            X_= X[idx]; y_ = y[idx]

            means_ = np.mean(X_, axis=0)
            X_ -= means_
            covariances_ = X_.T.dot(X_) / (X_.shape[0] - 1)

            self.prior_[i_class] = X_.shape[0] / X.shape[0]
            self.means_[i_class] = means_
            self.covariances_[i_class] = covariances_
        
    
    def predict(self, X):
        """ test estimator
        
        Params:
            X:      {ndarray(n_samples, n_feats)}
        Returns:
            y_pred: {ndarray(n_samples)}
        Notes:
            In case of overflow: 
                - use log(p(w_i|x)) instead.
        """
        
        n_samples = X.shape[0]
        n_class, n_feats = self.means_.shape

        ## calculate probability of each sample
        post_ = np.zeros(shape=(n_samples, n_class))
        for i_class in range(n_class):

            prior_ = self.prior_[i_class]
            means_ = self.means_[i_class]
            covariances_ = self.covariances_[i_class]
            
            for i_samples in range(n_samples):
                ## log post probability
                post_[i_samples, i_class] = self.logGaussian(X[i_samples], means_, covariances_) + np.log(prior_)

        ## get predicted label
        y_pred_proba = post_ - np.sum(post_, axis=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        return y_pred

    @classmethod
    def logGaussian(self, x, mean, covariances):
        """ log multiple Gaussian distribution
                            1                       1
        p(x) = ----------------------------- exp [ --- (x - \mu)^T \Sigma^{-1} (x - \mu) ]
                (2\pi)^{n/2} |\Sigma|^{1/2}         2
                    1                                       n                1
        log p(x) = --- (x - \mu)^T \Sigma^{-1} (x - \mu) - --- \log(2\pi) - --- \log(|\Sigma|) 
                    2                                       2                2

        Params:
            x:      {ndarray(n_samples, n_feats)}
            mean:   {ndarray(n_feats)}
            covariance: {ndarray(n_feats, n_feats)}
        Returns:
            px:     {ndarray(n_samples)}
        Notes:
            In case of overflow: 
                - use log(p(x)) instead.
        """
        n_feats = mean.shape[0]

        x -= mean
        covariances += np.eye(n_feats) * 1e-16
        inv_covar = np.linalg.inv(covariances)
        det_covar = np.linalg.det(covariances)

        a = 0.5 * x.T.dot(inv_covar).dot(x)
        b = (2*np.pi)**(n_feats/2) * np.sqrt(det_covar)

        # px = np.exp(a) / b
        logpx = a - np.log(b)
        
        return logpx


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)

    estimator = NaiveBayes()
    estimator.fit(X, y)

    y_pred = estimator.predict(X)

    acc = estimator.score(y, y_pred)
    print("accuracy score is {:>3.2f}%".format(acc*100))

    estimator.showfig(X, y)