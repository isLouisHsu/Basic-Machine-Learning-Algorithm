"""
问题：当设置n_batch=1时，算法收敛速度很慢。随机梯度下降(n_batch=-1)时算法收敛
"""
import numpy as np
import matplotlib.pyplot as plt

# Training set for the first half of the exercise - linear
# sigmoid(θ0*1 + θ1*x1 + θ2*x2)
# θ2*x2 = -θ0 - θ1*x1
file1 = './ex2data1.txt'
# Training set for the second half of the exercise - nonlinear
file2 = './ex2data2.txt'

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoidDerivative(x):
    return sigmoid(x)(1-sigmoid(x))

class LogisticRegression():
    def __init__(self, file, n_ploy=1, n_batch=1):
        '''
        n_batch: 取值为-1时为SGD，取值为1时为批处理梯度下降
        '''
        self.X, self.t = self.loadData(file, n_ploy=n_ploy, display=False)
        self.n_batch = self.X.shape[0] if n_batch==-1 else n_batch
        self.batch_size = self.X.shape[0] // self.n_batch   # 每一批数据的样本数
        self.theta = np.ones(shape=(self.X.shape[1],))      # 定义参数θ0,...,θk, 初始化为1
    def loadData(self, file, n_ploy=1, display=False):
        '''
        读取数据，添加全1列，返回X, y
        注：读取的数据都只有两个特征，故构造高次时只需考虑两列向量即可
        '''
        datalist = []
        with open(file) as f:
            dataline = f.readline()
            while(dataline):
                datalist.append(list(eval(dataline)))
                dataline = f.readline()
        dataarray = np.array(datalist)
        X = dataarray[:, :-1]; y = dataarray[:, -1]
        # 构造高次特征
        if n_ploy > 1:
            for i in range(n_ploy + 1):
                n1, n2 = n_ploy - i, i
                X = np.c_[X, (X[:, 0]**n1)*(X[:, 1]**n2)]
        # 加入全1列，便于计算参数
        X = np.c_[np.ones(shape=(X.shape[0]),), X]    
        if display==True:
            plt.figure('origin_data')
            plt.scatter(X[:, 1], X[:, 2], c=y)
            plt.show()
        return X, y
    def predict_prob(self, X, theta):
        '''
        对输入的样本进行预测
        X: NumPy array, X.shape=(n, k+1)
        theta: NumPy array, theta.shape=(k+1,)
        '''
        return sigmoid(X.dot(theta))
    def prob2Label(self, y_pred_prob):
        '''
        由predict_prob计算得概率，计算预测标签
        '''
        y_pred_label = np.sign(y_pred_prob - 0.5)
        y_pred_label[y_pred_label==-1] = 0
        return y_pred_label
    def accuracyRate(self, y_pred_prob, y_true):
        '''
        计算分类的准确率
        '''
        y_pred_label = self.prob2Label(y_pred_prob)
        correct = np.equal(y_pred_label, y_true)
        return np.sum(correct.astype('float'), axis=0)/y_true.shape[0]
    def lossFunction(self, y_pred_prob, y_true):
        '''
        未使用
        计算损失值: Cross-Entropy
        y_pred_prob, y_true: NumPy array, shape=(n,)
        '''
        tmp = y_true*np.log(y_pred_prob) + (1 - y_true)*np.log(1 - y_pred_prob)
        return np.mean(-tmp)
    def lossFunctionDerivative(self, X, theta, y_true):
        '''
        计算损失函数对参数theta的梯度
        对theta[j]的梯度为：(y_pred - y_true)*x[j]
        '''
        err = self.predict_prob(X, theta) - y_true
        return X.T.dot(err)/y_true.shape[0]
    def gradDescent(self, min_acc, learning_rate=0.01, max_iter=10000):
        '''
        利用梯度下降法求解参数
        '''
        acc = 0; n_iter = 0
        while n_iter<max_iter:
            n_iter += 1
            for n in range(self.n_batch):
                X_batch = self.X[n*self.batch_size:(n+1)*self.batch_size]
                t_batch = self.t[n*self.batch_size:(n+1)*self.batch_size]
                grad = self.lossFunctionDerivative(X_batch, self.theta, t_batch)
                self.theta -= learning_rate * grad # 梯度下降
                acc = self.accuracyRate(self.predict_prob(self.X, self.theta), self.t)
                if acc > min_acc:
                    print('第%d次迭代, 第%d批数据' % (n_iter, n))
                    print("当前总体样本准确率为: ", acc)
                    print("当前参数值为: ", self.theta)
                    return self.theta
            if n_iter%100 == 0:
                print('第%d次迭代' % n_iter)
                print('准确率： ', acc)
        print("超过迭代次数")
        print("当前总体样本准确率为: ", acc)
        print("当前参数值为: ", self.theta)
        return self.theta
    def showResult(self):
        '''
        显示图像
        '''
        # ---------------------
        plt.figure(0); plt.cla()
        plt.scatter(self.X[:, 1], self.X[:, 2], c=self.t)
        x1 = np.linspace(30, 100)
        x2 = -(self.theta[0] + self.theta[1]*x1)/self.theta[2]
        plt.plot(x1, x2, c='r')
        # ---------------------
        # plt.figure(1)
        # n = np.arange(1, self.t.shape[0]+1, 1)
        # y_pred_label = self.prob2Label(self.predict_prob(self.X, self.theta))
        # plt.scatter(n, self.t, c='b')
        # plt.scatter(n, y_pred_label, c='r')
        plt.show()

# -------------
mdl = LogisticRegression(file1, n_ploy=1, n_batch=-1)
mdl.gradDescent(min_acc=0.90, learning_rate=0.01, max_iter=50000)
mdl.showResult()
'''
第448次迭代, 第37批数据
当前总体样本准确率为:  0.92
当前参数值为:  [-29.12517994   0.25802142   0.21606811]
即边界方程为：-29.12517994 + 0.25802142*x1 + 0.21606811*x2 = 0
'''
# -------------
# mdl = LogisticRegression(file2, n_ploy=3, n_batch=-1)
# mdl.gradDescent(min_acc=0.85, learning_rate=0.01, max_iter=4000)
'''
n_ploy=2
    第1046次迭代, 第114批数据
    当前总体样本准确率为:  0.8559322033898306
    当前参数值为:  [ 4.01949941  2.36479865  3.13532241 -9.49346701 -5.46936331 -9.15036358]
    即边界方程为：4.01949941 + 2.36479865*x1 + 3.13532241*x2 -9.49346701*x1**2 - 5.46936331*x1*x2 - 9.15036358*x2**2  = 0
n_ploy=3
    超过迭代次数
    当前总体样本准确率为:  0.6694915254237288
    当前参数值为:  [ 0.40210341  3.31934137  4.9210368  -5.42726913 -8.1329048  -6.87297012  -7.62612016]
    即边界方程为：0.40210341 + 3.31934137*x1 + 4.9210368*x2 - 5.42726913*x1**3 - 8.1329048*x1**2*x2 - 6.87297012*x1*x2**2 - 7.62612016*x2**3 = 0
'''