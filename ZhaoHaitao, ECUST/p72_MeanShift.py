import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import sklearn.datasets as sd
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D

class MeanShift():
    # n_feature = 3
    def __init__(self, n_samples, n_features, k):
        """
        n：数据个数
        c：簇个数
        """
        self.k = k
        self.n_features = n_features
        self.X, self.y = sd.make_blobs(n_samples=n_samples, n_features=n_features, centers=k)
    def kernel(self, z):
        return np.exp(-0.5*np.square(nl.norm(z)))/np.sqrt(2*np.pi)
    def isInRange(self, x1, x2, r):
        if nl.norm(x1-x2)<=r:
            return True
        return False
    def chooseInitCenters(self, X, k):
        """
        该算法在2~3个簇时效果较好，5个及以上时效果一般
        X：样本
        c：簇的个数
        """
        n_features = self.n_features
        centers = np.zeros((k, n_features))
        cnt = 0
        while(cnt<k):
            # 第一个点
            if cnt==0:
                centers[0] = X[nr.randint(X.shape[0])]
                cnt += 1; continue
            # 后面的点：与已确定的中心点距离之和最远的点
            center = np.zeros((1, n_features));distMax = 0
            for i in range(X.shape[0]):
                dist = 0
                for j in range(cnt):
                    dist += nl.norm(X[i] - centers[j])
                if dist>distMax:
                    center  = X[i] 
                    distMax = dist
            centers[cnt] = center; cnt += 1
        # ---------------------------------------------------------
        # f = plt.figure()
        # if n_features==2:
        #     plt.scatter(self.X[:, 0], self.X[:, 1], c='b')
        #     plt.scatter(centers[:, 0], centers[:, 1], c='r')
        # else:
        #     ax = Axes3D(f)
        #     ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c='b')
        #     ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='r')
        # plt.show()
        # ---------------------------------------------------------
        return centers
    def meanShift(self, k, r=2, h=2, eps=0.01, n_iter=100):
        """
        c：簇的个数
        """
        n_features = self.n_features
        centers = self.chooseInitCenters(self.X, k)                 # 随机选择初始中心点
        while(n_iter>0):                                            # 最大迭代次数
            changed = False                                         # 标记位，中心点是否更新
            for i in range(centers.shape[0]):
                numerator = 0; denominator = 0
                for j in range(self.X.shape[0]):
                    if self.isInRange(self.X[j], centers[i], r):    # 判断是否再邻近范围内，这里采用球域
                        z = (self.X[j] - centers[i])/h
                        k = self.kernel(z)                          # 核函数：关于距离
                        numerator += self.X[j]*k                    # 质心公式的分子，self.X[j]为“密度”
                        denominator += k                            # 质心公式的分母
                centeri_new = numerator/denominator
                if nl.norm(centers[i] - centeri_new) > eps:
                    centers[i] = centeri_new; changed = True        # 大于阈值时更新中心点
            if not changed: break
            n_iter -= 1
            print(n_iter)
        # ---------------------------------------------------------
        # f = plt.figure()
        # if n_features==2:
        #     plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
        #     plt.scatter(centers[:, 0], centers[:, 1], c='r')
        # else:
        #     ax = Axes3D(f)
        #     ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=self.y)
        #     ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='r')
        # plt.show()
        # ---------------------------------------------------------
        return centers
    def predict(self, n_features):
        centers = self.meanShift(self.k)
        y = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            distMin = float('inf')
            y[i] = -1
            for j in range(centers.shape[0]):
                dist = nl.norm(self.X[i] - centers[j])
                if dist < distMin:
                    distMin = dist
                    y[i] = j            # 标为与最近的簇中心相同的类别
        # ---------------------------------------------------------
        f = plt.figure()
        if n_features==2:
            plt.scatter(self.X[:, 0], self.X[:, 1], c=y)
            plt.scatter(centers[:, 0], centers[:, 1], c='r')
        else:
            ax = Axes3D(f)
            ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=y)
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='r')
        plt.show()
        # ---------------------------------------------------------

if __name__ == '__main__':
    mdl = MeanShift(100, 3, 3)
    mdl.predict(3)