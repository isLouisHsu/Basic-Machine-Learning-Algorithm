import numpy as np
import numpy.linalg as nl
import sklearn.datasets as sd
import matplotlib.pyplot as plt

class KNN():
    def __init__(self, k, type):
        self.k = k
        self.type = type
        self.X, self.y, self.tX, self.ty = self.loadData()
    def loadData(self):
        # 共有3个种类别，每个类别50个样本。
        # 取出每个类别的前25个样本作为训练集，后25个样本作为测试集
        data, target = sd.load_iris(return_X_y=True)
        trainX = np.zeros(shape=(data.shape[0]//2, data.shape[1]))
        testX  = np.zeros(shape=(data.shape[0]//2, data.shape[1]))
        trainY = np.zeros(shape=target.shape[0]//2)
        testY  = np.zeros(shape=target.shape[0]//2)
        for i in range(3):  # 每个类别各取一半
            trainX[i*25: (i+1)*25] = data[i*50: i*50+25]
            testX[i*25: (i+1)*25]  = data[i*50+25: (i+1)*50]
            trainY[i*25: (i+1)*25] = target[i*50: i*50+25]
            testY[i*25: (i+1)*25]  = target[i*50+25: (i+1)*50]
        return trainX, trainY, testX, testY
    def dist(self, x1, x2, type='Euclidean'):
        if type=='Euclidean':   # 欧式距离
            return nl.norm(x1-x2)
        elif type=='cos':       # 余弦距离
            return x1.T.dot(x2)/(nl.norm(x1)*nl.norm(x2))
    def predict(self, x):
        # 对单个样本进行预测
        dst = np.zeros(self.X.shape[0])       # 与每个训练集样本的距离
        for i in range(self.X.shape[0]):
            dst[i] = self.dist(x, self.X[i], self.type)
        # 根据距离排列顺序
        # np.argsort(a):数组a从小到大排列的索引
        if self.type=='Euclidean':
            near = np.argsort(dst)[:self.k]        # 返回数组从小到大排列的索引，选取最小的k个
        elif self.type=='cos':
            near = np.argsort(dst)[-1:-self.k:-1]  # 返回数组从小到大排列的索引，选取最大的k个
        # 根据最近距离判断所属类别
        # np.bincount(a):返回一个数组，其索引值为a中的元素，其索引值对应的值为a中元素出现的次数
        # np.argmax(a):返回数组a中最大元素的索引
        for idx in range(near.shape[0]):           
            near[idx] = self.y[near[idx]]          # 索引对应为所属类别
        return np.argmax(np.bincount(near))        # 返回类别出现次数最多的作为该样本的类别
    def predictAll(self):
        y = np.zeros(self.tX.shape[0])
        for i in range(self.tX.shape[0]):
            y[i] = self.predict(self.tX[i])
        return y
    def showResult(self):
        predictY = self.predictAll()
        figureX = np.arange(0, self.tX.shape[0])
        plt.figure()
        plt.axvline(25);plt.axvline(50)             # 做一条切分竖线
        plt.scatter(figureX, predictY, c=self.ty)   # 绘制散点图
        plt.show()


if __name__ == '__main__':
    mdl_Euc = KNN(15, 'Euclidean')
    mdl_Euc.showResult()
    mdl_cos = KNN(15, 'cos')        # 效果更好
    mdl_cos.showResult()