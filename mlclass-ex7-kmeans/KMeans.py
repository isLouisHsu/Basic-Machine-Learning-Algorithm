import numpy as np
import matplotlib.pyplot as plt
import load_data
import mathFunc

class KMeans():
    def __init__(self, n_cluster, mode):
        self.n_cluster = n_cluster  # 簇的个数
        self.mode = mode            # 距离度量方式
        self.centroids = None       # 簇的中心
        self.loss = float('inf')    # 优化目标值
        plt.ion()
    def fit(self, X, max_iter=50, min_epison=0.1):
        def initializeCentroids():
            '''
            选择初始点
            '''
            centroid = np.zeros(shape=(self.n_cluster, X.shape[1])) # 保存选出的点
            pointIdx = []                                           # 保存已选出的点的索引
            for n in range(self.n_cluster):
                idx = np.random.randint(0, X.shape[0])              # 随机选择一个点
                while idx in pointIdx:                              # 若该点已选出，则丢弃重新选择
                    idx = np.random.randint(0, X.shape[0])
                pointIdx.append(idx)
                centroid[n] = X[idx]
            return centroid
        def dist2Centroids(x, mode):
            '''
            返回向量x到k个中心点的距离值
            '''
            d = np.zeros(shape=(self.n_cluster,))
            for n in range(self.n_cluster):
                d[n] = mathFunc.distance(x, self.centroids[n], mode)
            return d
        def nearestInfo(mode):
            '''
            每个点最近的簇中心索引、距离
            '''
            ctIdx = -np.ones(shape=(X.shape[0],), dtype=np.int8)    # 每个点最近的簇中心索引，初始化为-1，可作为异常条件
            ctDist = np.ones(shape=(X.shape[0],), dtype=np.float)   # 每个点到最近簇中心的距离
            for i in range(X.shape[0]):
                dists = dist2Centroids(X[i], mode)
                if mode == 'Euclidean':
                    ctIdx[i] = np.argmin(dists)
                elif mode == 'Cosine':
                    ctIdx[i] = np.argmax(dists)
                ctDist[i] = dists[ctIdx[i]]             # 保存最相似的距离度量，用于计算loss
            return ctIdx, ctDist
        def updateCentroids(ctIdx):
            '''
            更新簇中心
            '''
            centroids = np.zeros(shape=self.centroids.shape)
            for n in range(self.n_cluster):
                X_ = X[ctIdx == n]                      # 筛选出离簇中心Cn最近的样本点
                centroids[n] = np.mean(X_, axis=0)      # 根据筛选出的样本点更新中心值
            return centroids
        def loss(dist):
            return np.mean(dist**2)
        # -----------------------------------------
        self.centroids = initializeCentroids()          # 选择初始点
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            ctIdx, ctDist = nearestInfo(mode=self.mode) # 每个点到最近的簇中心相关信息：索引、距离值
            # --- 可视化 ---
            if X.shape[1] == 2:
                plt.figure(0); plt.cla()
                plt.scatter(X[:, 0], X[:, 1], c=ctIdx)
                plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='r')
                plt.pause(0.5)
            # -------------
            self.centroids = updateCentroids(ctIdx)     # 更新簇中心
            loss_current = loss(ctDist)
            if np.abs(self.loss - loss_current) < min_epison:
                print('变化太小')
                break
            self.loss = loss_current
        if n_iter >= max_iter:
            print('超过最大迭代次数')
        print('当前损失值为:', self.loss)

    def predict(self, X):
        '''
        各个样本的最近簇中心索引
        '''
        labels = -np.ones(shape=(X.shape[0],), dtype=np.int) # 初始化为-1，可用作异常条件
        for i in range(X.shape[0]):
            dists_i = np.zeros(shape=(self.n_cluster,))
            for n in range(self.n_cluster):
                dists_i[n] = mathFunc.distance(X[i], self.centroids[n], mode=self.mode)
            if self.mode == 'Euclidean':
                labels[i] = np.argmin(dists_i)
            elif self.mode == 'Cosine':
                labels[i] = np.argmax(dists_i)
        return labels

def chooseBestK(X, start, stop, step=1, mode='Euclidean'):
    Ks = np.arange(start, stop + 1, step, dtype=np.int) # 待选择的K
    Losses = np.zeros(shape=Ks.shape)                   # 保存不同K值时的最小损失值
    for k in range(1, Ks.shape[0] + 1):                 # 对于不同的K，训练模型，计算损失
        estimator = KMeans(n_cluster=k, mode=mode)
        estimator.fit(X)
        Losses[k - 1] = estimator.loss
    plt.ioff()
    plt.figure()
    plt.plot(Ks, Losses)                                # 做出loss-K曲线
    plt.show()
    
    
if __name__ == '__main__':
    X = load_data.load_2D(load_data.data_kmeans)


    chooseBestK(X, start=1, stop=5, step=1, mode='Euclidean')
    best_K = eval(input('please type in the best K: '))
    # ====> the best K is 3
    estimator = KMeans(n_cluster=best_K, mode='Euclidean')
    estimator.fit(X, max_iter=500, min_epison=0.1)


    chooseBestK(X, start=1, stop=5, step=1, mode='Cosine')
    best_K = eval(input('please type in the best K: '))
    # ====> the best K is 3
    estimator = KMeans(n_cluster=best_K, mode='Cosine')
    estimator.fit(X, max_iter=500, min_epison=0.1)

    