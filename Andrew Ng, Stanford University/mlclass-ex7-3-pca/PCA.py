import numpy as np
import matplotlib.pyplot as plt
from load_data import data_pca, load_2D

class PrincipalComponentAnalysis():
    def __init__(self, n_component=-1):
        self.n_component = n_component
        self.meanVal = None
        self.varVal  = None
        self.axis = None
    def fit(self, X, prop=0.99):
        '''
        the parameter 'prop' is only for 'n_component = -1'
        '''
        # 第一步: 归一化
        self.meanVal = np.mean(X, axis=0)                   # 训练样本每个特征上的的均值
        self.varVal  = np.var(X, axis=0)                    # 训练样本每个特征上的的方差
        X_normalized = (X - self.meanVal) / self.varVal     # 归一化训练样本
        # 第二步：计算协方差矩阵
        # cov = X_normalized.T.dot(X)
        cov = np.cov(X_normalized.T)                        # 协方差矩阵
        eigVal, eigVec = np.linalg.eig(cov)                 # EVD
        order = np.argsort(eigVal)[::-1]                    # 从大到小排序
        eigVal = eigVal[order]
        eigVec = eigVec.T[order].T
        # 选择主成分的数量
        if self.n_component == -1:
            sumOfEigVal = np.sum(eigVal)
            sum_tmp = 0
            for k in range(eigVal.shape[0]):
                sum_tmp += eigVal[k]
                if sum_tmp > prop * sumOfEigVal:            # 平均均方误差与训练集方差的比例尽可能小的情况下选择尽可能小的 K 值
                    self.n_component = k + 1
                    break
        # 选择投影坐标轴
        self.axis = eigVec[:, :self.n_component]            # 选择前n_component个特征向量作为投影坐标轴
    def transform(self, X):
        # 第一步：归一化
        X_normalized = (X - self.meanVal) / self.varVal     # 归一化测试样本
        # 第二步：投影 X_nxk · V_kxk' = X'_nxk'
        X_transformed = X_normalized.dot(self.axis)
        return X_transformed
    def fit_transform(self, X, prop=0.99):
        self.fit(X, prop=prop)
        return self.transform(X)
    def transform_inv(self, X_transformed):
        # 视投影向量长度为一个单位长度，投影结果为投影向量上的坐标
        # X'_nxk' · V_kxk'.T = X''_nxk
        X_restructed = X_transformed.dot(self.axis.T)
        # 还原数据
        X_restructed = X_restructed * self.varVal + self.meanVal
        return X_restructed

if __name__ == '__main__':
    X = load_2D(data_pca, display=True)

    reduceDim = PrincipalComponentAnalysis(n_component=-1)
    reduceDim.fit(X, prop=0.5)
    X_transfromed = reduceDim.transform(X)

    plt.figure()
    
    # plt.scatter(np.arange(X_transfromed.shape[0]), X_transfromed[:, 0])
    # plt.savefig('./data/reduced_1.png')     # 协方差矩阵用np.cov()求解 
    # plt.savefig('./data/reduced_2.png')     # 协方差矩阵用X.T.dot(X)求解
                                            # 两个结果在图形上显示一致

    X_restructed = reduceDim.transform_inv(X_transfromed)
    plt.scatter(X[:, 0], X[:, 1], c='b')
    plt.scatter(X_restructed[:, 0], X_restructed[:, 1], c='r')

    plt.show()
    ############################################################