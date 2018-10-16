import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import matplotlib.pyplot as plt
import sklearn.datasets as skds
import sklearn.cluster as skclus

def loadData():
    n_clusters = 2
    # 上半圆方程±0.5.随机数
    x = np.arange(-5, 5, 0.02)
    y = np.sqrt(25 - x**2) 
    x = x + nr.rand(x.shape[0]) - 0.5
    y = y + nr.rand(y.shape[0]) - 0.5 # nr.rand()生成(0,1)间的随机数，减去0.5是去均值化(-0.5,0.5)
    X1 = np.c_[x, y]
    n1 = X1.shape[0]
    y1 = np.zeros(n1)
    # 中心二维正态分布
    mu = np.array([0, 2])
    s = np.array([[1, 0], [0, 1]])
    n2 = 100
    X2 = np.dot(np.random.randn(n2, 2), nl.cholesky(s)) + mu
    y2 = np.ones(n2)
    # 合并为数据集
    X = np.r_[X1, X2]
    y = np.r_[y1, y2]
    # 显示数据
    # plt.scatter(X[:,0], X[:,1], c=y); plt.show()
    return X, y, n_clusters
def loadIris():
    n_clusters = 3
    X, y = skds.load_iris(True)
    return X, y, n_clusters

class SpectralClustering():
    def __init__(self, data, sigma, c):
        self.X, self.t, self.n_clusters = data()
        self.y = self.spectralClustering(self.X, self.n_clusters, sigma, c)
        self.evaluate(self.t, self.y)
    def spectralClustering(self, X, n_clusters, sigma, c):
        '''
        n_clusters: 聚类个数
        c：取拉普拉斯矩阵最大的c个特征值和对应特征向量
        '''
        # 高斯核函数
        def kernelGaussian(z, sigma):
            return np.exp(-0.5*np.square(z/sigma))
        # 相似度矩阵(全连接)
        W = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(i):
                W[i, j] = kernelGaussian(nl.norm(X[i]-X[j]), sigma)
                W[j, i] = W[i, j]
        # 度矩阵
        D = np.diag(np.sum(W, axis=1))
        # 拉普拉斯矩阵
        # --未正则化
        L = D - W
        # --正则化
        # L = nl.inv(D).dot(L)
        # 求拉普拉斯矩阵的特征值和对应特征向量
        lda, vec = nl.eig(L)
        # 取前c列作为新矩阵(未进行正交初始化)
        V = vec[:, 0:c]
        # 对V的行向量进行KMeans聚类
        clf = skclus.KMeans(n_clusters)
        clf.fit(V)
        return clf.labels_
    def evaluate(self, t, y):
        # 准确度评估
        # 有问题，分类正确，标签不一定相同
        N = t.shape[0]
        e = t - y
        n = e[np.where(e==0)].shape[0]
        print("准确度：", 100*n/N, "%")

if __name__ == "__main__":
    # print("-----------------------------------------------------------------")
    # print("改变c")
    # model = SpectralClustering(loadData, sigma=0.4, c=2)    # 准确度： 2.1666666666666665 %
    # model = SpectralClustering(loadData, sigma=0.4, c=3)    # 准确度： 83.5 %
    # model = SpectralClustering(loadData, sigma=0.4, c=4)    # 准确度： 98.5 %
    # model = SpectralClustering(loadData, sigma=0.4, c=5)    # 准确度： 83.5 %
    # print("改变sigma")
    # model = SpectralClustering(loadData, sigma=0.2, c=2)    # 准确度： 83.16666666666667 %
    # model = SpectralClustering(loadData, sigma=0.3, c=2)    # 准确度： 84.0 %
    # model = SpectralClustering(loadData, sigma=0.4, c=2)    # 准确度： 83.16666666666667 %
    # model = SpectralClustering(loadData, sigma=0.5, c=2)    # 准确度： 2.0 %
    print("-----------------------------------------------------------------")
    print("改变c")
    model = SpectralClustering(loadIris, sigma=0.4, c=2)    # 准确度： 32.0 %
    model = SpectralClustering(loadIris, sigma=0.4, c=3)    # 准确度： 32.0 %
    model = SpectralClustering(loadIris, sigma=0.4, c=4)    # 准确度： 32.0 %
    model = SpectralClustering(loadIris, sigma=0.4, c=5)    # 准确度： 32.0 %
    print("改变sigma")
    model = SpectralClustering(loadIris, sigma=0.2, c=2)    # 准确度： 32.0 %
    model = SpectralClustering(loadIris, sigma=0.3, c=2)    # 准确度： 32.0 %
    model = SpectralClustering(loadIris, sigma=0.4, c=2)    # 准确度： 32.0 %
    model = SpectralClustering(loadIris, sigma=0.5, c=2)    # 准确度： 0.0 %
