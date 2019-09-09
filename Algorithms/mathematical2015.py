# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-07-17 18:07:29
@LastEditTime: 2019-08-11 11:24:54
@Update: 
'''
import os
import cv2
import numpy as np
import pickle
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
# from 

"""
|-- 2015年全国研究生数学建模竞赛B题
    |-- 数据
        |-- 第1题数据
            |-- 1.mat
        |-- 第2题数据
            |-- 2a.mat
            |-- 2b.mat
            |-- 2c.mat
            |-- 2d.mat
        |-- 第3题数据
            |-- 3a.mat
            |-- 3b.mat
            |-- 3c.mat
        |-- 第4题数据
            |-- 4a.mat
            |-- 4b.mat
"""
data_path        = './2015年全国研究生数学建模竞赛B题/数据'
exercise1_fname  = '第1题数据/1.mat'
exercise2a_fname = '第2题数据/2a.mat'
exercise2b_fname = '第2题数据/2b.mat'
exercise2c_fname = '第2题数据/2c.mat'
exercise2d_fname = '第2题数据/2d.mat'
exercise3a_fname = '第3题数据/3a.mat'
exercise3b_fname = '第3题数据/3b.mat'
exercise3c_fname = '第3题数据/3c.mat'
exercise4a_fname = '第4题数据/4a.mat'
exercise4b_fname = '第4题数据/4b.mat'

def load_mat(filename):
    mat = loadmat(filename)
    data = mat['data'].T
    return data
def show2dfig(X, labels=None):
    if X.shape[1] != 2: print('dimension error!'); return
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, marker='.')
    plt.show()
def show3dfig(X, labels=None):
    if X.shape[1] != 3: print('dimension error!'); return
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, marker='.')
    plt.show()

def exercise1(X):
    reduce_dim = PCA(n_components=2)
    X_reduced = reduce_dim.fit_transform(X)
    X_reduced = X_reduced.astype('float')

    estimator = KMeans(n_clusters=2)
    estimator.fit(X_reduced)
    labels = estimator.predict(X_reduced)
    
    show2dfig(X_reduced, labels)

def exercise2a(X):
    n_samples = X.shape[0]
    show2dfig(X)

    # 对数据进行重建
    reduce_dim = PCA(n_components=2)
    X = reduce_dim.fit_transform(X)
    show2dfig(X)

    y = np.zeros(shape=(n_samples))

    # index = np.abs(X[:, 0]) < 0.15
    # y[index] = 1
    # show2dfig(X, y)

    hyperparaboloid = lambda x, y: x**2 - y**2	# 双曲抛物面/马鞍面
    z = hyperparaboloid(X[:, 0], X[:, 1])
    y[z > 0] = 1
    show2dfig(X, y)

    return y

def exercise2b(X):
    n_samples = X.shape[0]
    y = np.zeros(shape=(n_samples,))
    # show3dfig(X, y)

    # 对数据进行重建
    reduce_dim = PCA(n_components=3)
    X = reduce_dim.fit_transform(X)
    # show3dfig(X)

    # 将重构后的数据投影到第1、2主成分张成的平面spam{v1, v2}上
    # 这里用到了线性回归的投影解释
    v = reduce_dim.axis[:, 0:2]
    norm_res = np.zeros(shape=(n_samples,))
    for i in range(n_samples):
        y_ = X[i]
        w = np.linalg.pinv(v).dot(y_)       # 公式求解最优回归系数 w* = (X^TX)^{-1} X^T y
        y__ = v.dot(w)
        # 计算与1、2主成分张成平面垂直的向量，这里记作res
        res = y__ - y_
        norm_res[i] = np.linalg.norm(res)
    # 取均值为阈值
    thresh = np.mean(norm_res)
    # 比阈值小即在平面内，标记为3
    index = (norm_res<=thresh)
    y[index] = 2
    # show3dfig(X, y)

    # 剩余的点作pca投影到2维平面，用2a求解
    X_rest = X.copy()
    X_rest[index] = np.array([0, 0, 0])
    X_rest = reduce_dim.fit_transform(X_rest)
    y[~index] = exercise2a(X_rest)[~index]

    show3dfig(X, y)
    return y

def exercise2c(X):
    n_samples = X.shape[0]
    y = np.zeros(shape=(n_samples,))
    show3dfig(X, y)

    # 对数据进行重建
    reduce_dim = PCA(n_components=2)
    X = reduce_dim.fit_transform(X)
    show2dfig(X, y)

    # 广义线性回归拟合两类点之间的曲线
    ploy = PolynomialFeatures(degree=2)     # 构造 0, 1, 2次特征

    x1 = X[:, 0].reshape((-1, 1))
    x1_ = ploy.fit_transform(x1)             
    x2_ = X[:, 1]
    w = np.linalg.pinv(x1_).dot(x2_)

    # 显示
    # plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c='b')
    # x1__ = np.linspace(np.min(x1), np.max(x1))
    # x2__ = ploy.transform(x1__.reshape((-1, 1))).dot(w)
    # plt.plot(x1__, x2__, c='r')
    # plt.show()
    
    y[x2_ > x1_.dot(w)] = 1
    show2dfig(X, y)
    return y

def exercise2d(X):
    pass

def exercise3a():
    pass
def exercise3b():
    pass
def exercise3c():
    pass

def exercise4a(X):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    y = np.zeros(shape=(n_samples,))
    # show3dfig(X, y)

    def norm_vec():
        """ 计算每个点切平面的法向量，运算量大，保存结果
        """
        # 建立数据的KDTree
        kdtree = KDTree(X, metric='euclidean')
        vecs = np.zeros(shape=(n_samples, n_features))
        I = np.eye(n_features); y_ = np.mean(X, axis=0)
        for i in range(n_samples):
            # 10个近邻点组成近似平面
            X_i = X[i].reshape((1, -1))     # Reshape your data either using array.
                                            # reshape(-1, 1) if your data has a single feature or array.
                                            # reshape(1, -1) if it contains a single sample.
            idx = kdtree.query(X_i, 10-1, return_distance=False)
            X_nn = X[idx][0]
            pca = PCA(n_components=n_features)
            pca.fit(X_nn)
            vecs[i] = pca.components_[-1]   # 最次的分量为法向量
        np.save("./data/vec.npy", vecs)

    def angle(x, y):
        cosval = x.dot(y) / (np.linalg.norm(x)*np.linalg.norm(y))
        angval = np.arccos(np.clip(cosval, -1, 1)) * 180 / np.pi
        return angval
    def mark(k=10, d=0.2, a=5):
        vecs = np.load("./data/vec.npy")
        # vecs = loadmat('./2015年全国研究生数学建模竞赛B题/Xs10.mat')['X1'].T

        points = np.c_[X, np.zeros(n_samples)]       # 最后一列用于表示类别
        Xrest = np.c_[X, np.zeros(n_samples), np.arange(n_samples)] # -1列对数据进行标号，表示与原数据的映射，-2列记录数据所属类别
        nowMax = 1
        while Xrest[Xrest[:, -2]==0].shape[0] > 0:   # 判断剩余矩阵的个数
            restRate = Xrest[Xrest[:, 3]==0].shape[0] / n_samples
            print("剩余{:.2%}未标记".\
                    format(restRate))
            nowId = 0
            x0 = Xrest[nowId, :3]
            v0 = vecs[int(Xrest[nowId, -1])]        # 直接从 pca 结果中提取法向量
            while True:
                # 搜索d近邻内的点
                kdtree = KDTree(Xrest[:, :3], metric='euclidean')
                k_ = Xrest.shape[0] if Xrest.shape[0] < k else k
                dist, idx = kdtree.query(x0.reshape((1, -1)), k=k_)
                dist = dist.reshape(-1); idx = idx.reshape(-1)
                idx = idx[dist<d]
                # 筛选角度小于阈值的点
                for i in range(idx.shape[0]):
                    idx_ = idx[i]
                    v1 = vecs[int(Xrest[idx_, 4])]
                    ang = angle(v0, v1)
                    if (ang < a) or (ang > 180 - a):
                        points[int(Xrest[idx_, -1]), -1] = nowMax
                        Xrest[idx_, -2] = nowMax
                # 移除当前点
                Xrest = np.delete(Xrest, nowId, axis=0)
                # 搜索同类别下一个点
                if Xrest[Xrest[:, 3]==nowMax].shape[0] > 0:
                    for i in range(Xrest.shape[0]):
                        if Xrest[i, -2] == nowMax:
                            nowId = i
                            x0 = Xrest[i, :3]
                            v0 = vecs[int(Xrest[i, -1])]
                            break
                else:
                    nowMax += 1
                    break

        # 统计各类别样本点的个数
        cnt = np.bincount(points[:, -1].astype('int'))  # 各标签计数
        mainmark = np.argsort(cnt)[::-1]                # 按从多到少排序
        # 标记各点
        labels = points[:, -1].astype('int')
        idx_1 = (labels==mainmark[0])                   # 确定为1的点的索引
        idx_2 = (labels==mainmark[1])                   # 确定为2的点的索引
        idx_3 = (labels==mainmark[2])                   # 确定为3的点的索引
        points[idx_1, -1] = 1
        points[idx_2, -1] = 2
        points[idx_3, -1] = 3
        idx_0 = ~(idx_1 + idx_2 + idx_3)                # 未确定标记的点索引
        points[idx_0, -1] = 0                           # 将其标签置为0

        # 将剩余的点归入最多的3个簇中
        X_marked = np.c_[points, np.arange(n_samples)]  # 增加与原数据的映射列
        # 建立KDTree
        kdtree = KDTree(X_marked[:, :3], metric='euclidean')
        while (X_marked[:, -2]==0).shape[0] > 0:
            Xrest = X_marked[X_marked[:, -2]==0]        # 提取出这些点
            n_rest = Xrest.shape[0]
            print('{:2d} points left'.format(n_rest))
            if n_rest == 0: break
            idx_rest = np.random.randint(0, n_rest)
            idx = kdtree.query(Xrest[idx_rest, :3].reshape((1, -1)), 
                        k=20, return_distance=False)    # 查询最近的k_个点
            idx_labels = X_marked[idx.reshape(-1), -2].astype('int')
            cnt = np.bincount(idx_labels)               # 最近点所属类别计数
            if cnt.shape[0] == 1: continue
            X_marked[int(Xrest[idx_rest, -1]), -2] = np.argmax(cnt[1:]) + 1
        return X_marked[:, -2].astype('int')

    
    # 对数据进行重建
    reduce_dim = PCA(n_components=3)
    X = reduce_dim.fit_transform(X)
    # show3dfig(X)

    # norm_vec()
    y = mark()
    show3dfig(X[y!=0], y[y!=0])
    pass
    


    

def exercise4b():
    pass



if __name__ == '__main__':
    # X = load_mat(os.path.join(data_path, exercise1_fname)); exercise1(X)

    # X = load_mat(os.path.join(data_path, exercise2a_fname)); exercise2a(X)
    # X = load_mat(os.path.join(data_path, exercise2b_fname)); exercise2b(X)
    # X = load_mat(os.path.join(data_path, exercise2c_fname)); exercise2c(X)

    X = load_mat(os.path.join(data_path, exercise4a_fname)); exercise4a(X)