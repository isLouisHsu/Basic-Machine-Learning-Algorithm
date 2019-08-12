'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-08-09 17:11:31
@LastEditTime: 2019-08-12 10:26:07
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


class NeighborhoodPreservingEmbedding():
    """ Neighborhood Preserving Embedding

    Attributes:
        n_neighbors:  {int}
        n_components: {int}
        W_: {ndarray} 
        components_:    {ndarray(n_samples, n_components)}
    """
    def __init__(self, n_neighbors, n_components=2, k_skip=1):
        
        self.n_neighbors  = n_neighbors
        self.n_components = n_components
        self.k_skip = k_skip

        self.W_ = None
        self.components_ = None

    def fit(self, X):
        """ 
        Params:
            X: {ndarray(n_samples, n_features)}
        """
        from sklearn.neighbors import KDTree
        kdtree = KDTree(X, metric='euclidean')
        
        n_samples, n_features = X.shape
        self.W_ = np.zeros((n_samples, n_samples))
        for i in range(n_samples):

            ## 获取近邻样本点
            x = X[i]
            idx = kdtree.query(x.reshape(1, -1), self.n_neighbors + 1, return_distance=False)[0][1: ]
            ## 求取矩阵 Z = (x - N).dot((x - N).T)
            N = X[idx]
            Z = (x - N).dot((x - N).T)
            ## 求取权重 w_i
            Z_inv = np.linalg.inv(Z + np.finfo(float).eps * np.eye(self.n_neighbors))
            w = np.sum(Z_inv, axis=1) / np.sum(Z_inv)
            ## 保存至 W
            for j in range(self.n_neighbors):
                self.W_[idx[j], i] = w[j]
        
        ## 求取矩阵 M = (I - W)(I - W)^T
        I = np.eye(n_samples)
        M = (I - self.W_).dot((I - self.W_).T)

        ## 求解 X M X^T \alpha = \lambda X X^T \alpha
        A1 = X.T.dot(M).dot(X)
        A2 = X.T.dot(X)
        eps = np.finfo(float).eps * np.eye(A2.shape[0])
        A = np.linalg.inv(A2 + eps).dot(A1)

        ## 对 A 进行特征分解，并按特征值升序排序
        eigval, eigvec = np.linalg.eig(A)
        eigvec = eigvec[:, np.argsort(eigval)]
        eigval = eigval[np.argsort(eigval)]
        
        ## 选取 D 维
        self.components_ = eigvec[:, self.k_skip: self.n_components + self.k_skip]

    def transform(self, X):
        """ 
        Params:
            X: {ndarray(n_samples, n_features)}
        Returns:
            Y: {ndarray(n_samples, n_components)}
        """
        Y = X.dot(self.components_)
        
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


if __name__ == "__main__":

    from sklearn import datasets
    
    # -----------------------------------------------------------------------------
    digits = datasets.load_digits(n_class=6)
    X = digits.data
    y = digits.target
    images = digits.images

    npe = NeighborhoodPreservingEmbedding(30, 2, k_skip=3)
    X_npe = npe.fit_transform(X)

    plot_embedding(X_npe, y, images, title=None, t=2e-3, figsize=(12, 9))

    # -----------------------------------------------------------------------------
    X, color = datasets.samples_generator.make_s_curve(1000, random_state=0)
    
    npe = NeighborhoodPreservingEmbedding(10, 2)
    X_npe = npe.fit_transform(X)

    plt.figure()
    plt.scatter(X_npe[:, 0], X_npe[:, 1], c=color)
    plt.show()