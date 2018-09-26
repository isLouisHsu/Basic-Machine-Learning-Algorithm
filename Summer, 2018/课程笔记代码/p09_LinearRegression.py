import numpy as np
import numpy.linalg as nl

class LinearRegression():
    def __init__(self, m, n):
        self._m = m         # m组数据
        self._n = n         # 每组n维
        self.creatData()
    def creatData(self):
        self.X = np.c_[np.ones((self._m, 1)), np.random.uniform(-5, 5, size=(self._m, self._n))]   # m组数据，每组d0个特征
        self.W = np.reshape(np.array([10*(i+1) for i in range(self._n+1)]), (self._n+1, 1))
        self.T = self.H_(self.W) + np.random.normal(scale=2.0)
    def getWTheoretical(self):
        # θ=[(XTX)-1]XTy
        return nl.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.T)
    def H_(self, W):
        # 输出函数
        return np.dot(self.X, W)
    def J(self, W):
        # 目标函数
        E = self.H_(W) - self.T
        return 0.5*np.sum(E**2)
    def gradJ(self, W):
        # 目标函数的梯度
        E = self.H_(W) - self.T
        return np.dot(self.X.T, E)
    def getWGradDecent(self, threshold):
        W = np.ones((self._n+1, 1))
        alpha = 0.001               # 这里没有对学习率进行优化
        while self.J(W)>threshold:  # 迭代
            W -= alpha*self.gradJ(W)
        return W

if __name__ == "__main__":
    LR = LinearRegression(10, 5)
    J = LR.J(LR.W)
    W = LR.getWGradDecent(1.001*J)
    print(W)
    print(LR.getWTheoretical())

            