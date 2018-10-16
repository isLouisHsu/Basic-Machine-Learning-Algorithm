import numpy as np
import numpy.linalg as nl
import sklearn.datasets as sd
import sklearn.decomposition as sde
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PCA():
    '''
    注：PCA降维后不用考虑坐标变换带来的数据方向问题，都是正确的
    课程笔记代码中的数据集没有，故用iris数据集代替
    '''

    def __init__(self):
        self.X, self.y = sd.load_iris(True)

    def pca(self, k):
        # 每个维度去均值化
        means = np.mean(self.X, axis=0)
        X = self.X - means
        # 计算协方差矩阵
        cov = self.X.T.dot(self.X)
        # cov = np.cov(self.X.T)
        # 计算协方差矩阵的特征向量和特征值
        w, v = nl.eig(cov)
        # 选择2个特征值最大的维度进行投影，得到表示在v系的坐标点
        s = self.X.dot(v[:, :k])
        return s

    def pca_sklearn(self, k):
        return sde.PCA(n_components=k).fit_transform(self.X)


if __name__ == '__main__':
    mdl = PCA()
    # 降成2维数据
    S1 = mdl.pca(2)
    plt.figure(0); plt.title('by handwriting')
    plt.scatter(S1[:,0], S1[:,1], c=mdl.y)
    S2 = mdl.pca_sklearn(2)
    plt.figure(1); plt.title('by sklearn')
    plt.scatter(S2[:,0], S2[:,1], c=mdl.y)
    
    # 聚类不明显，降成3维显示
    # S1 = mdl.pca(3)
    # f = plt.figure()
    # ax = Axes3D(f)
    # ax.scatter(S1[:,0], S1[:,1], S1[:,2], c=mdl.y)
    plt.show()
