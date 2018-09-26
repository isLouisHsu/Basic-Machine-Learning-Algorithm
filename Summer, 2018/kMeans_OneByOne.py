'''
逐个加入训练样本的方式
'''
import numpy as np
from numpy import random as nr
import matplotlib.pyplot as plt

def distMinkowski(x, y, p):
    '''
    x,y: np.array (shape[0], 1)
    '''
    sum = 0
    for i in range(x.shape[0]):
        sum += (x[i]-y[i])**p
    return sum**(1/p)

def idOfMinInList(list):
    min = list[0]
    idx = 0
    for i in range(len(list)):
        if list[i] < min:
            min = list[i]
            idx = i
    return idx

color = ['r', 'g', 'y', 'gray', 'lightpink', 'lightseagreen', 'linen', 'lime', 'b']

class KMeans():
    def __init__(self, k, m, n):
        self._k = k
        self._m = m     # m组数据
        self._n = n     # 每组n维
        self.X = np.reshape(np.array([]), (0, self._n))
        for i in range(self._k):
            self.X = np.r_[self.X, nr.uniform(i*3, (i+1)*3, size=(int(self._m/self._k), self._n))]
        self.initMeans = self.chooseInitMeans()                             # 选取k个初始中心点
        self.listClassify = [np.reshape(self.initMeans[i], (1, self._n)) for i in range(self._k)]   # 用于存储各个分类样本
        self.means = self.initMeans
    def chooseInitMeans(self, random=False):
        # 选择k个初始中心点，初始中心点位置对分类结果有影响
        means = np.zeros((self._k, self._n))
        if random==True:
            id = nr.randint(self._m, size=(self._k, 1))
            for i in range(self._k):
                x = np.reshape(self.X[i], (1, self._n))
                means[i] = x
        else:
            for i in range(self._k):
                x = np.reshape(np.array([i*4, (i+1)*4]), (1, self._n))
                means[i] = x
        return means
    def classify(self):
        for i in range(self.X.shape[0]):
            print(i)
            x = self.X[i]
            listDist = []
            for k in range(self._k):
                listDist.append(distMinkowski(x, self.means[k], 2))
            id = idOfMinInList(listDist)
            self.listClassify[id] = np.r_[self.listClassify[id], np.reshape(x, (1, self._n))]
            self.updateMeans()

            # if self._n==2:
            #     plt.figure()
            #     x1 = self.X[:,0]
            #     x2 = self.X[:,1]
            #     plt.scatter(x1, x2, c='b')
            #     for j in range(self._k):
            #         x1 = self.listClassify[j][:,0]
            #         x2 = self.listClassify[j][:,1]
            #         plt.scatter(x1, x2, c=color[j+1])
            #     # x1 = self.initMeans[:,0]
            #     # x2 = self.initMeans[:,1]
            #     # plt.scatter(x1, x2, c='r')
            #     x1 = self.means[:,0]
            #     x2 = self.means[:,1]
            #     plt.scatter(x1, x2, c='r')
            #     plt.savefig(".\\kmeans结果\\image"+str(i)+".png")

        # if self._n==2:
        #     plt.figure()
        #     for i in range(self._k):
        #         x1 = self.listClassify[i][:,0]
        #         x2 = self.listClassify[i][:,1]
        #         plt.scatter(x1, x2, c=color[i+1])
        #     # x1 = self.initMeans[:,0]
        #     # x2 = self.initMeans[:,1]
        #     # plt.scatter(x1, x2, c='r')
        #     x1 = self.means[:,0]
        #     x2 = self.means[:,1]
        #     plt.scatter(x1, x2, c='r')
        #     plt.show()
    def updateMeans(self):
        for i in range(self._k):
            meansofdata = self.meansOfData(self.listClassify[i])
            self.means[i] = np.reshape(meansofdata, (self._n,))
        pass
    def meansOfData(self, data):
        ret = np.zeros((1, data.shape[1]))
        for j in range(data.shape[1]):
            ret[0, j] = np.sum(data[:, j])/data.shape[0]
        return ret
if __name__ == '__main__':
    KM = KMeans(4, 48, 2)
    KM.classify()
