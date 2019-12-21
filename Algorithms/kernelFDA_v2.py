# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-05 11:49:34
@LastEditTime : 2019-12-21 20:05:29
@Update: 
'''
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

class KernelFDA():

    def __init__(self, n_components, kernel, gamma=None, degree=3, coef0=1.):

        self.n_components_ = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        self.samples_    = None
        self.targets_    = None
        self.n_samples_  = None
        self.n_features_ = None

        self.components_ = None

    def _kernelMatrix(self, X, Y=None):
        """ Calculate kernel matrix
        Params:
            X: {ndarray(n_samples, n_features)}
        Returns:
            K:  {ndarray(n_samples, n_samples)} the kernel matrix
        """
        params = {"gamma": self.gamma,
                "degree": self.degree,
                "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y):
        """
        Params:
            X: {ndarray(n_samples, n_features)}
            y: {ndarray(n_samples)}
        """
        labels = list(set(list(y)))
        n_classes = len(labels)
        n_samples, n_features = X.shape

        self.samples_ = X
        self.targets_ = y
        self.n_samples_  = n_samples
        self.n_features_ = n_features
        
        K = self._kernelMatrix(X)
        
        M = np.zeros((n_samples, n_samples))
        N = np.zeros((n_samples, n_samples))
        mean_ = np.mean(K, axis=0)
        for i_class in range(n_classes):
            K_ = K[y==labels[i_class]]
            
            means_ = np.mean(K_, axis=0)

            K_ = K_ - means_
            means_ = (means_ - mean_).reshape(1, -1)

            M += (K_.T).dot(K_) * (1 / n_samples)
            N += (means_.T).dot(means_) * (K_.shape[0] / n_samples)

        s, u = np.linalg.eigh(M)
        s[s <= 0] = 1e-8
        s_sqrt = np.diag(np.sqrt(s))
        s_sqrt_inv = np.linalg.inv(s_sqrt)

        A = s_sqrt_inv.dot(u.T).dot(N).dot(u).dot(s_sqrt_inv)
        eigval, P = np.linalg.eigh(A)
        eigvec = u.dot(s_sqrt_inv).dot(P)

        order = np.argsort(eigval)[::-1]
        eigval = eigval[order]
        eigvec = eigvec[:, order]

        self.components_ = eigvec[:, :self.n_components_]

    def transform(self, X):
        """
        Params:
            X: {ndarray(n_samples, n_features)}
        Returns:
            Y: {ndarray(n_samples, n_components)}
        Notes:
            y_k(x) = \Phi(x)^T v_k \sum_{i=1}^N a_{ki} \kappa(x, x_i)
        """
        K = self._kernelMatrix(X, self.samples_)
        Y = K.dot(self.components_)

        return Y

    def transform_inv(self, X):

        raise NotImplementedError(
                    "Inverse transformation is not implemented")
    
    def fit_transform(self, X, y):
        """
        Params:
            X: {ndarray(n_samples, n_features)}
            y: {ndarray(n_samples)}
        Returns:
            Y: {ndarray(n_samples, n_components)}
        """

        self.fit(X, y)
        return self.transform(X)

if __name__ == "__main__":
    
    from sklearn.datasets import make_circles
    from matplotlib import pyplot as plt

    X, y = make_circles(n_samples=400, factor=.3, noise=.05)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)

    kpca = KernelFDA(2, 'rbf')
    Y = kpca.fit_transform(X, y)

    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c=y)
    plt.show()
