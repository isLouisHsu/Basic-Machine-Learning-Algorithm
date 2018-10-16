import numpy as np
import sklearn.datasets as sd
import matplotlib.pyplot as plt

class NaiveBays():
    def __init__(self):
        self.X, self.y, self.tX, self.ty = self.loadData()
        self.mu, self.sigma = self.train(self.X)
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
    def normal(self, x, mu, sigma):
        return np.exp(-0.5*np.square((x-mu)/sigma))/(np.sqrt(2*np.pi)*sigma)
    def train(self, X):
        # 假设每个维度都服从正态分布
        # 训练的数据为P(x|ci)
        # 计算每个类别各个维度上的均值μ和标准差σ
        n_class = 3     # 类别数
        mu = np.zeros((n_class, X.shape[1]))
        sigma = np.zeros((n_class, X.shape[1]))
        for c in range(n_class):                           # 各个类别
            for i in range(X.shape[1]):                    # 各个维度
                mu[c, i] = np.mean(X[c*25:(c+1)*25, i])    # 均值
                sigma[c, i] = np.std(X[c*25:(c+1)*25, i])  # 标准差
        return mu, sigma
    def predict(self, x):
        #           P(x|ci)P(ci)
        # P(ci|x) = ————————————
        #               P(x)
        # 其中，P(ci|x)为后验概率(post)；P(x|ci)为ci关于x的似然函数(prodt)
        # 用朴素贝叶斯的思路，P(x|ci) = ΠP(xj|ci) P(x) = ΠP(xj)
        # 这里使用正态分布概率密度函数的值代替概率P(xj|ci)
        # 由于输入样本x固定，故P(x)视为常数不做计算
        # 可使用softmax
        n_class = 3     # 类别数
        post = np.zeros(n_class)
        for c in range(n_class):
            prodt = 1
            for i in range(self.X.shape[1]):
                prodt *= self.normal(x[i], self.mu[c, i], self.sigma[c, i])   # 朴素贝叶斯
            pc = 1.0/3          # P(ci)的值已知，若未知则需统计
            post[c] = prodt*pc  # 在此处可添加风险因子
        # softmax
        # softmaxPost = np.zeros(n_class)
        # denominator = 0
        # for c in range(n_class):
        #     denominator += np.exp(post[c])
        # for c in range(n_class):
        #     softmaxPost[c] = np.exp(post[c])/denominator
        return np.argmax(post)  # 返回最大值索引，对应类别
    def predictAll(self, X):
        y = np.zeros(self.ty.shape)
        for i in range(X.shape[0]):
            y[i] = self.predict(X[i])
        return y
    def showResult(self):
        predY = self.predictAll(self.tX)
        figureX = np.arange(0, self.tX.shape[0])
        plt.figure()
        plt.subplot(221)
        plt.title('feature1')
        # plt.xlabel('n_sample')
        plt.axvline(25);plt.axvline(50)
        plt.scatter(figureX, self.tX[:, 0], c=predY)
        plt.subplot(222)
        plt.title('feature2')
        # plt.xlabel('n_sample')
        plt.axvline(25);plt.axvline(50)
        plt.scatter(figureX, self.tX[:, 1], c=predY)
        plt.subplot(223)
        plt.title('feature3')
        # plt.xlabel('n_sample')
        plt.axvline(25);plt.axvline(50)
        plt.scatter(figureX, self.tX[:, 2], c=predY)
        plt.subplot(224)
        plt.title('feature4')
        # plt.xlabel('n_sample')
        plt.axvline(25);plt.axvline(50)
        plt.scatter(figureX, self.tX[:, 3], c=predY)
        plt.show()
if __name__ == '__main__':
    mdl = NaiveBays()
    mdl.showResult()
    