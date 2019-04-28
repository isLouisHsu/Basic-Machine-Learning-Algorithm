import numpy as np
from matplotlib import pyplot as plt

class LDA(object):
    """ 先行判别分析

    Attributes:
        n_components:   {int} 主成分个数
        axis:           {ndarray(n_features, n_component)}
    Notes:
        S_W = \frac{1}{m} \sum_{j=1}^C \sum_{i=1}^m_j (x^{(i)} - \mu^{(j)}) (x^{(i)} - \mu^{(j)})^T
        S_B = \sum_{j=1}^C \frac{m_j}{m} (\mu^{(j)} - \mu) (\mu^{(j)} - \mu)^T
    Example:

    """

    def __init__(self, n_components=-1):
        self.n_components = n_components
        self.axis = None

    def fit(self, X, y, prop=0.99):
        """ train the model
        Params:
            X:      {ndarray(n_samples, n_features)}
            y:      {ndarray(n_samples)}
            prop:   {float}:  在[0, 1]范围内，表示取特征值之和占所有特征值的比重
        Notes:
            - `prop`参数仅在`n_components=-1`时生效
        """
        labels = list(set(list(y)))
        n_class = len(labels)
        n_samples, n_feats = X.shape

        ## 计算 S_W, S_B
        S_W = np.zeros(shape=(n_feats, n_feats))
        S_B = np.zeros(shape=(n_feats, n_feats))
        mean_ = np.mean(X, axis=0)
        for i_class in range(n_class):
            X_ = X[y==labels[i_class]]
            weight = X_.shape[0] / n_samples
            means_ = np.mean(X_, axis=0)
            S_W += ((X_ - means_).T).dot(X_ - means_) / X_.shape[0] * weight
            S_B += (means_ - mean_).dot((means_ - mean_).T) * weight

        ## 计算特征对
        eigVal, eigVec = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        order = np.argsort(eigVal)[::-1]
        eigVal = eigVal[order]
        eigVec = eigVec.T[order].T

        ## 选取主轴
        if self.n_components == -1:
            sumOfEigVal = np.sum(eigVal)
            sum_tmp = 0
            for k in range(eigVal.shape[0]):
                sum_tmp += eigVal[k]
                if sum_tmp > prop * sumOfEigVal:
                    self.n_components = k + 1
                    break
        self.axis = eigVec[:, :self.n_components]

    def transform(self, X):
        """
        Params:
            X:  {ndarray(n_samples, n_features)}
        Returns:
            X:  {ndarray(n_samples, n_components)}
        """
        X = X.dot(self.axis)
        return X
    
    def fit_transform(self, X, y, prop=0.99):
        """
        Params:
            X:  {ndarray(n_samples, n_features)}
        Returns:
            X:  {ndarray(n_samples, n_components)}
        """
        self.fit(X, y, prop=prop)
        X = self.transform(X)
        return X
    
    def transform_inv(self, X):
        """
        Params:
            X:  {ndarray(n_samples, n_components)}
        Returns:
            X:  {ndarray(n_samples, n_features)}
        """
        X = X.dot(self.axis.T)
        return X

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from p50_pca import PCA
    
    X, y = load_iris(True)

    decomposer = LDA(n_components=2)
    X1 = decomposer.fit_transform(X, y)
    decomposer = PCA(n_component=2)
    X2 = decomposer.fit_transform(X, y)

    plt.figure("LDA")
    plt.scatter(X1[:, 0], X1[:, 1], c=y)
    plt.figure("PCA")
    plt.scatter(X2[:, 0], X2[:, 1], c=y)
    plt.show()
