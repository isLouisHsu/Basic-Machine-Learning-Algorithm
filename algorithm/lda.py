# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-07-17 18:07:29
@LastEditTime: 2019-09-05 20:15:18
@Update: 
'''
import numpy as np
from matplotlib import pyplot as plt

import numpy as np

class LDA(object):
    """ 

    Attributes:
        n_components: {int}
        means_: {ndarray(n_classes, n_features)}
        components_:  {ndarray(n_components, n_features)}
    """

    def __init__(self, n_components):
        self.n_components = n_components

        self.means_ = None
        self.components_ = None

    def fit(self, X, y):
        """ train the model
        Params:
            X:      {ndarray(n_samples, n_features)}
            y:      {ndarray(n_samples)}
        """
        labels = list(set(list(y)))
        n_classes = len(labels)
        n_samples, n_features = X.shape

        self.means_ = np.zeros((n_classes, n_features))
        S_W = np.zeros(shape=(n_features, n_features))
        S_B = np.zeros(shape=(n_features, n_features))
        mean_ = np.mean(X, axis=0)
        for i_class in range(n_classes):
            X_ = X[y==labels[i_class]]
            
            means_ = np.mean(X_, axis=0)
            self.means_[i_class] = means_

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
        self.components_ /= np.linalg.norm(self.components_, axis=1).reshape(1, -1)

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

    def predict(self, X):
        """
        Params:
            X:  {ndarray(n_samples, n_features)}
        Returns:
            y:  {ndarray(n_samples)}
        """
        n_samples, n_features = X.shape
        y = np.zeros(n_samples, dtype=np.int)
        
        X_ = self.transform(X)
        means_ = self.transform(self.means_)

        for i in range(n_samples):
            y[i] = np.argmin(np.linalg.norm(means_ - X_[i], axis=1))
        
        return y

if __name__ == "__main__":

    # # ====================================================
    # from sklearn.datasets import load_iris
    # from p50_pca import PCA
    
    # X, y = load_iris(True)

    # decomposer = LDA(n_components=2)
    # X1 = decomposer.fit_transform(X, y)
    # decomposer = PCA(n_components=2)
    # X2 = decomposer.fit_transform(X, y)

    # plt.figure("LDA")
    # plt.scatter(X1[:, 0], X1[:, 1], c=y)
    # plt.figure("PCA")
    # plt.scatter(X2[:, 0], X2[:, 1], c=y)
    # plt.show()
    
    # # ====================================================
    # from matplotlib import pyplot as plt
    # from sklearn.datasets import make_blobs

    # X, y = make_blobs(n_samples=[200, 200])

    # plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    # clf = LDA(n_components=1)
    # clf.fit(X, y)

    # y_pred = clf.predict(X)

    # plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    # plt.show()
    
    # ====================================================
    from matplotlib import pyplot as plt
    from p50_pca import PCA
    
    n_samples_per_class = 10000
    t = np.pi / 4

    M = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    x_ = np.random.normal(loc=0.0, scale=50.0, size=n_samples_per_class)
    y_ = np.random.normal(loc=0.0, scale= 1.0, size=n_samples_per_class)
    xy1 = np.r_[x_, y_].reshape(2, -1).T
    xy1 = xy1.dot(M)

    x_ = np.random.normal(loc=0.0, scale=50.0, size=n_samples_per_class)
    y_ = np.random.normal(loc=0.0, scale= 1.0, size=n_samples_per_class)
    xy2 = np.r_[x_, y_].reshape(2, -1).T
    xy2 = xy2.dot(M)

    X1 = xy1 + np.array([0,  0]); y1 = np.zeros(n_samples_per_class)
    X2 = xy2 + np.array([0, 20]); y2 = np.ones (n_samples_per_class)
    X  = np.r_[X1, X2]; y = np.r_[y1, y2]


    lda = LDA(n_components=1)
    X1 = lda.fit_transform(X, y)
    pca = PCA(n_components=1)
    X2 = pca.fit_transform(X, y)

    plt.figure(figsize=(8, 8))

    plt.scatter(X[::25, 0], X[::25, 1], c=y[::25])

    xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
    f1 = lambda x: lda.components_[0, 1] / lda.components_[0, 0] * x
    f2 = lambda x: pca.components_[1, 0] / pca.components_[0, 0] * x
    plt.plot([xmin, xmax], [f1(xmin), f1(xmax)], c='red')
    plt.plot([xmin, xmax], [f2(xmin), f2(xmax)], c='blue')

    plt.show()