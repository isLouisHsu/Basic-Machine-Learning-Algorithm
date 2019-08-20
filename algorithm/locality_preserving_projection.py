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
        from sklearn import neighbors

        ## 计算邻接矩阵
        self.W_ = neighbors.kneighbors_graph(X, self.n_neighbors, 
                mode=self.mode, metric=self.metric).toarray()
        
        ## 计算权值矩阵
        self.W_ = np.exp(- self.W_ / self.t)
        
        ## 计算度矩阵与拉普拉斯矩阵
        D = np.diag(np.sum(self.W_, axis=1))
        L = D - self.W_

        ## 求解拉普拉斯矩阵
        A1 = X.T.dot(L).dot(X)
        A2 = X.T.dot(D).dot(X)
        eps = np.finfo(float).eps * np.eye(A2.shape[0])
        A  = np.linalg.inv(A2 + eps).dot(A1)

        ## 求解拉普拉斯矩阵的特征分解
        eigval, eigvec = np.linalg.eig(A)
        eigvec = eigvec[:, np.argsort(eigval)[::-1]]
        
        ## 选取主分量
        self.components_ = eigvec[:, :self.n_components]

    def fit(self, X):
        """
        Params:
            X: {ndarray(n_samples, n_features)}
        """
        self._fit_transform()
    
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
