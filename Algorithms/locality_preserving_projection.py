# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-08-11 15:21:23
@LastEditTime: 2019-08-20 10:10:34
@Update: 
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox

from utils import eig

def plot_embedding(X, y, images, title=None, t=6e-3, figsize=(12, 9)):
    """ Plot embedding
    
    Params:
        X: {ndarray(N, n_features)}
        y: {ndarray(N)}
        images: {ndarray(N, H, W)}
        title: {str}
        t: {float} threshold
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < t:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(
                    images[i], cmap=plt.cm.gray_r), X[i])
            ax.add_artist(imagebox)
            
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
        
    plt.show()

class LocalityPreservingProjection():
    """ Locality Preserving Projection

    Params:
        n_neighbors:    {int}
        t:              {float}
        mode:           {str} 'distance', 'connectivity'
        metric:         {str} 'euclidean', 'minkowski'
    
    Attributes:
        W_:             {ndarray(n_samples, n_samples)}
        components_:    {ndarray(n_samples, n_components)}
    """

    def __init__(self, n_neighbors, n_components=2,  t=3., mode='distance', metric='euclidean'):

        self.n_neighbors  = n_neighbors
        self.n_components = n_components
        self.t = t
        self.mode = mode
        self.metric = metric

        self.W_ = None
        self.components_ = None
        
    def _fit_transform(self, X):
        """
        Params:
            X: {ndarray(n_samples, n_features)}
        Notes:
            $$ W_{ij} = \exp \left( - \frac{||x^{(i)} - x^{(j)}||_2^2}{t} \right) $$
        """
        ## 计算邻接矩阵
        from sklearn.neighbors import KDTree
        kdtree = KDTree(X, metric='euclidean')
        
        n_samples, n_features = X.shape
        self.W_ = np.zeros((n_samples, n_samples))
        for i in range(n_samples):

            ## 获取近邻样本点
            xi = X[i]
            idx = kdtree.query(xi.reshape(1, -1), self.n_neighbors + 1, return_distance=False)[0][1: ]
            for j in idx:
                xj = X[j]
                self.W_[i, j] = \
                self.W_[j, i] = \
                    np.linalg.norm(xi - xj)
        
        ## 计算权值矩阵
        self.W_ = np.where(self.W_ != 0, np.exp(- self.W_**2 / self.t), self.W_)
        
        ## 计算度矩阵与拉普拉斯矩阵
        D = np.diag(np.sum(self.W_, axis=1))
        I = np.eye(X.shape[0])
        L = D - self.W_
        
        ## 求解拉普拉斯矩阵
#         A1 = X.T.dot(L).dot(X)
#         A2 = X.T.dot(D).dot(X)
#         eigval, eigvec = eig(A1, A2)
#         eigvec = eigvec[:, np.argsort(eigval)]
        
        ## 求解正则拉普拉斯矩阵
        D_r = np.linalg.inv(np.sqrt(D))
        L = D_r.dot(L).dot(D_r)
        A1 = X.T.dot(I - L).dot(X)
        A2 = X.T.dot(X) + np.diag(np.ones(X.shape[1])) * 1e-3
        
        eigval, eigvec = eig(A1, A2)
        eigvec = eigvec[:, np.argsort(eigval)[::-1]]
        
        ## 选取主分量
        self.components_ = eigvec[:, :self.n_components]

    def fit(self, X):
        """
        Params:
            X: {ndarray(n_samples, n_features)}
        """
        self._fit_transform(X)
    
    def transform(self, X):
        """
        Params:
            X: {ndarray(n_samples, n_features)}
        """
        Y = X.dot(self.components_)

        return Y

    def fit_transform(self, X):
        """
        Params:
            X: {ndarray(n_samples, n_features)}
        """
        self._fit_transform(X)
        return self.transform(X)


if __name__ == "__main__":

    from sklearn import datasets
    
    # -----------------------------------------------------------------------------
    digits = datasets.load_digits(n_class=6)
    X = digits.data
    y = digits.target
    images = digits.images

    lpp = LocalityPreservingProjection(30, 2)
    X_lpp = lpp.fit_transform(X)

    plot_embedding(X_lpp, y, images, title=None, t=2e-3, figsize=(12, 9))

    # -----------------------------------------------------------------------------
    X, color = datasets.samples_generator.make_s_curve(1000, random_state=0)
    
    lpp = LocalityPreservingProjection(10, 2)
    X_lpp = lpp.fit_transform(X)

    plt.figure()
    plt.scatter(X_lpp[:, 0], X_lpp[:, 1], c=color)
    plt.show()
