# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-05 11:49:23
@LastEditTime: 2019-09-05 20:36:40
@Update: 
'''
import numpy as np

class KernelPCA():
    """ Kernel Principal Component Analysis

    Attributes:
        n_components_:  {int} number of components
        kernel_:        {callable function}
        samples_:       {ndarray(n_samples, n_features)}
        n_samples_:     {int} number of samples
        n_features_:    {int} number of features
        components_:    {ndarray(n_samples, n_components)}

    Example:
        ```
        from kernel import Kernel
        from sklearn.datasets import make_circles
        from matplotlib import pyplot as plt

        X, y = make_circles(n_samples=400, factor=.3, noise=.05)
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y)

        kernel=lambda x, y: Kernel.gaussian(x, y, 0.5)
        kpca = KernelPCA(2, kernel)

        Y = kpca.fit_transform(X)

        plt.figure()
        plt.scatter(Y[:, 0], Y[:, 1], c=y)
        plt.show()
        ```
    """
    def __init__(self, n_components, kernel):

        self.n_components_ = n_components
        self.kernel_ = kernel
        
        self.samples_    = None
        self.n_samples_  = None
        self.n_features_ = None

        self.components_ = None

    def _kernelMatrix(self, X):
        """ Calculate kernel matrix
        Params:
            X: {ndarray(n_samples, n_features)}
        Returns:
            K1: {ndarray(n_samples, n_samples)} the kernel matrix
            K2: {ndarray(n_samples, n_samples)} the gram matrix
        Notes:
            $$ K2 = K1 - 1_N K - K 1_N + 1_N K 1_N $$
            where $1_N$ is the $N \times N$ matrix with all elements equal to 1/N.
        """
        n_samples, _ = X.shape
        
        K1 = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                K1[i, j] = K1[j, i] = self.kernel_(X[i], X[j])

        I  = np.ones_like(K1) / n_samples
        K2 = K1 - I.dot(K1) - K1.dot(I) + I.dot(K1).dot(I)
        
        return K1, K2

    def fit(self, X):
        """
        Params:
            X: {ndarray(n_samples, n_features)}
        """
        n_samples, n_features = X.shape

        self.samples_ = X
        self.n_samples_, self.n_features_ = n_samples, n_features

        _, K2 = self._kernelMatrix(X)

        eigval, eigvec = np.linalg.eigh(K2)
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
        n_samples, _ = X.shape
        
        K = np.zeros((n_samples, self.n_samples_))
        for i in range(n_samples):
            for j in range(self.n_samples_):
                K[i, j] = self.kernel_(X[i], self.samples_[j])
        
        Y = K.dot(self.components_)

        return Y

    def transform_inv(self):

        raise NotImplementedError(
                    "Inverse transformation is not implemented")
    
    def fit_transform(self, X):
        """
        Params:
            X: {ndarray(n_samples, n_features)}
        Returns:
            Y: {ndarray(n_samples, n_components)}
        """

        self.fit(X)
        return self.transform(X)

if __name__ == "__main__":
    
    from kernel import Kernel
    from sklearn.datasets import make_circles
    from matplotlib import pyplot as plt

    X, y = make_circles(n_samples=400, factor=.3, noise=.05)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)

    kernel=lambda x, y: Kernel.gaussian(x, y, 0.5)
    kpca = KernelPCA(2, kernel)

    Y = kpca.fit_transform(X)

    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c=y)
    plt.show()