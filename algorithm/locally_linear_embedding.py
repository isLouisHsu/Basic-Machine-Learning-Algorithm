'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-08-09 17:11:31
@LastEditTime: 2019-08-11 11:12:07
@Update: 
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox

from sklearn.datasets  import load_digits
from sklearn.neighbors import KDTree

class LocallyLinearEmbedding():
    """ Locally Linear Embedding

    Attributes:
        n_neighbors:  {int}
        n_components: {int}
        W: {ndarray} 
            $$ W = \left[ \begin{matrix} w_1 & w_2 & \cdots & w_{n_samples} \end{matrix} \right] $$
            $$ w_i = \left[ \begin{matrix} w_{i1} & w_{i2} & \cdots & w_{i, n_samples} \end{matrix} \right]^T $$
    """
    def __init__(self, n_neighbors, n_components=2):
        
        self.n_neighbors  = n_neighbors
        self.n_components = n_components
        self.W = None

    def fit(self, X):
        """ 
        Params:
            X: {ndarray(n_samples, n_features)}
        Returns:
            W: {ndarray(n_samples, n_samples)}
        """
        n_samples, n_features = X.shape
        self.W = np.zeros((n_samples, n_samples))

        kdtree = KDTree(X, metric='euclidean')
        for i in range(n_samples):

            ## 获取近邻样本点
            x = X[i]
            idx = kdtree.query(x.reshape(1, -1), self.n_neighbors + 1, return_distance=False)[0][1: ]

            ## 求取矩阵 Z = (x - N).dot((x - N).T)
            N = X[idx]
            Z = (x - N).dot((x - N).T)

            ## 求取权重 w_i
            Z_inv = np.linalg.inv(Z + 1e-16*np.eye(self.n_neighbors))
            w = np.sum(Z_inv, axis=1) / np.sum(Z_inv)
            
            ## 保存至 W
            for j in range(self.n_neighbors):
                self.W[i, idx[j]] = w[j]
        
        self.W = self.W.T
        return self.W

    def transform(self, X):
        """ 
        Params:
            X: {ndarray(n_samples, n_features)}
        Returns:
            Y: {ndarray(n_samples, n_components)}
        """
        n_samples, n_features = X.shape

        ## 求取矩阵 A = (I - W)(I - W)^T
        I = np.eye(n_samples)
        A = (I - self.W).dot((I - self.W).T)

        ## 对 A 进行特征分解，并按特征值升序排序
        eigval, eigvec = np.linalg.eig(A)
        eigvec = eigvec[:, np.argsort(eigval)]

        ## 选取 D 维
        Y = eigvec[:, 1: self.n_components + 1]

        return Y

    def fit_transform(self, X):
        """ 
        Params:
            X: {ndarray(n_samples, n_features)}
        Returns:
            Y: {ndarray(n_samples, n_components)}
        """
        
        self.fit(X)
        Y = self.transform(X)

        return Y
    

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


if __name__ == "__main__":
    
    digits = load_digits(n_class=6)
    X = digits.data
    y = digits.target
    images = digits.images

    n_samples, n_features = X.shape
    n_neighbors = 30

    lle = LocallyLinearEmbedding(30, 2)
    X_lle = lle.fit_transform(X)

    plot_embedding(X_lle, y, images, title=None, t=2e-3, figsize=(12, 9))


