# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-07-17 18:07:29
@LastEditTime: 2019-08-20 15:37:25
@Update: 
'''
import numpy as np
from matplotlib import pyplot as plt

class LDA(object):
    """ Linear Discriminant Analysis 

    Attributes:
        n_components: {int}
        components_:  {ndarray(n_components, n_features)}
    Notes:
    -   S_W = \frac{1}{m} \sum_{j=1}^C \sum_{i=1}^m_j (x^{(i)} - \mu^{(j)}) (x^{(i)} - \mu^{(j)})^T
    -   S_B = \sum_{j=1}^C \frac{m_j}{m} (\mu^{(j)} - \mu) (\mu^{(j)} - \mu)^T
    -   S_W^{-1} S_B \alpha = \lambda \alpha
    -   由于特征分解时，计算出现复数，故进行相应处理，详情查看https://louishsu.xyz/2019/04/22/LDA/
    Example:

    """

    def __init__(self, n_components=-1):
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, y):
        """ train the model
        Params:
            X:      {ndarray(n_samples, n_features)}
            y:      {ndarray(n_samples)}
        """
        labels = list(set(list(y)))
        n_class = len(labels)
        n_samples, n_feats = X.shape

        S_W = np.zeros(shape=(n_feats, n_feats))
        S_B = np.zeros(shape=(n_feats, n_feats))
        mean_ = np.mean(X, axis=0)
        for i_class in range(n_class):
            X_ = X[y==labels[i_class]]
            means_ = np.mean(X_, axis=0)

            X_ = X_ - means_
            means_ = (means_ - mean_).reshape(1, -1)

            S_W += (X_.T).dot(X_) * (1 / n_samples)
            S_B += (means_.T).dot(means_) * (X_.shape[0] / n_samples)

        s, u = np.linalg.eigh(S_W)
        s_sqrt = np.diag(np.sqrt(s))
        s_sqrt_inv = np.linalg.inv(s_sqrt)

        A = s_sqrt_inv.dot(u.T).dot(S_B).dot(u).dot(s_sqrt_inv)
        eigval, P = np.linalg.eigh(A)
        eigvec = u.dot(s_sqrt_inv).dot(P)

        order = np.argsort(eigval)[::-1]
        eigval = eigval[order]
        eigvec = eigvec[:, order]

        self.components_ = eigvec[:, :self.n_components].T

    def transform(self, X):
        """
        Params:
            X:  {ndarray(n_samples, n_features)}
        Returns:
            X:  {ndarray(n_samples, n_components)}
        """
        X_ = X.dot(self.components_.T)
        return X_
    
    def fit_transform(self, X, y):
        """
        Params:
            X:  {ndarray(n_samples, n_features)}
        Returns:
            X:  {ndarray(n_samples, n_components)}
        """
        self.fit(X, y)
        X_ = self.transform(X)
        return X_
    
    def transform_inv(self, X):
        """
        Params:
            X:  {ndarray(n_samples, n_components)}
        Returns:
            X:  {ndarray(n_samples, n_features)}
        """
        X_ = X.dot(self.components_)
        return X_

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from p50_pca import PCA
    
    X, y = load_iris(True)

    decomposer = LDA(n_components=2)
    X1 = decomposer.fit_transform(X, y)
    decomposer = PCA(n_components=2)
    X2 = decomposer.fit_transform(X, y)

    plt.figure("LDA")
    plt.scatter(X1[:, 0], X1[:, 1], c=y)
    plt.figure("PCA")
    plt.scatter(X2[:, 0], X2[:, 1], c=y)
    plt.show()
