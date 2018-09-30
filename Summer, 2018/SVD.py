import numpy as np
import numpy.linalg as nl
import sklearn.datasets as sd
import sklearn.decomposition as sde
import matplotlib.pyplot as plt

class SVD():
    def __init__(self):
        self.X = sd.load_iris().data
    def svd(self):
        # 如何解决XTX不正定的问题：XTX一定正定
        XTX = np.dot(self.X.T, self.X)
        XXT = np.dot(self.X, self.X.T)
        # 计算V
        s2, V = nl.eig(XTX)
        # 计算σ
        s = np.sqrt(s2)
        # 计算U
        # U = np.zeros((self.X.shape[0], self.X.shape[0]))
        # for i in range(V.shape[1]):
        #     U[:, i] = self.X.dot(V[:, i])/s[i]
        tmp, U = nl.eig(XXT)                    # ！注意，这里出现的复数形式，数量级在e-13，可忽略不计，视作0
        # 验证正确性
        sigma = np.r_[np.diag(s), np.zeros((self.X.shape[0]-self.X.shape[1], self.X.shape[1]))]
        E = U.dot(sigma).dot(V.T)-self.X
        print(E)
        print(s)
        
    def svd_sklearn(self, n_components):
        return sde.TruncatedSVD(n_components).fit_transform(self.X)

if __name__ == '__main__':
    mdl = SVD()
    mdl.svd()
    # print(mdl.svd_sklearn(2))
