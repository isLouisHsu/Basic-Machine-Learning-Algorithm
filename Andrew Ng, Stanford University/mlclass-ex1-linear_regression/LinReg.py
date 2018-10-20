import numpy as np
import matplotlib.pyplot as plt

# Linear regression with one variable
# y = θ0 + θ1*x
file1 = './ex1data1.txt'
# Linear regression with multiple variables
# y = θ0 + θ1*x1 + θ2*x2
file2 = './ex1data2.txt'

class LinearRegression():
    def __init__(self, file, n_batch=1):
        '''
        n_batch: 取值为-1时为SGD，取值为1时为批处理梯度下降
        '''
        self.X, self.t = self.loadData(file)
        self.n_batch = self.X.shape[0] if n_batch==-1 else n_batch
        self.batch_size = self.X.shape[0] // self.n_batch   # 每一批数据的样本数
        self.theta = np.ones(shape=(self.X.shape[1],))      # 定义参数θ0,...,θk, 初始化为1
        # --------- 可视化相关 ---------
        self.visualizeEnable = False
        plt.ion()                                           # 开启interactive mode
        if self.X.shape[1] == 2:                            # 已加上全1列
            self.visualizeEnable = True
            plt.xlim(0, 25); plt.ylim(-5,25)                # 设置画布大小
        # -----------------------------
    def loadData(self, file):
        datalist = []
        with open(file) as f:
            dataline = f.readline()                     # 读取一行数据(字符串)
            while dataline:
                datalist.append(list(eval(dataline)))   # 保存到2维list
                dataline = f.readline()
        dataarray = np.array(datalist)                  # list转换为array
        X_y = np.c_[np.ones(shape=(dataarray.shape[0]),), dataarray]    # 加入全1列，便于计算参数
        X = X_y[:, :-1]; y = X_y[:, -1]
        # 进行尺度缩放, 注意, 若输入测试集数据，需保存max_和min_进行相同的尺度缩放
        # for i in range(1, X.shape[1]):
        #     max_ = np.max(X[:, i]); min_ = np.min(X[:, i])
        #     X[:, i] = (X[:, i] - min_)/(max_-min_)
        return X, y                                     # return X, y
    def predict(self, X, theta):
        '''
        对输入的样本进行预测
        X: NumPy array, X.shape=(n, k+1)
        theta: NumPy array, theta.shape=(k+1,)
        '''
        return X.dot(theta)
    def lossFunction(self, y_pred, y_true):
        '''
        计算损失值: MSE
        y_pred, y_true: NumPy array, shape=(n,)
        '''
        err = y_pred - y_true
        return np.mean(0.5*np.square(err))
    def lossFunctionDerivative(self, X, theta, y_true):
        '''
        计算损失函数对参数theta的梯度
        对theta[j]的梯度为：(y_pred - y_true)*x[j]
        '''
        y_pred = self.predict(X, theta)
        grad = X.T.dot(y_pred - y_true)/X.shape[0]              # 遍历用矩阵运算代替, !注意: 这里需求一次均值
        # grad = self.theta = np.zeros(shape=(self.X.shape[1],))
        # for j in range(self.X.shape[1]): grad[j] = np.mean(err.dot(self.X[:, j]))
        return grad
    def fit(self, min_loss, learning_rate=0.01, max_iter=500):
        '''
        利用梯度下降法求解参数
        '''
        n_iter = 0
        while(n_iter<max_iter):
            n_iter += 1
            for n in range(self.n_batch):
                X_batch = self.X[n*self.batch_size:(n+1)*self.batch_size]
                t_batch = self.t[n*self.batch_size:(n+1)*self.batch_size]
                grad = self.lossFunctionDerivative(X_batch, self.theta, t_batch)
                self.theta -= learning_rate*grad # 梯度下降
                loss = self.lossFunction(self.predict(self.X, self.theta), self.t)  # 计算全部样本的损失函数值
                # self.visualization(self.visualizeEnable & (n_iter%300==0))          # 可视化
                if loss < min_loss:
                    print('第%d次迭代, 第%d批数据' % (n_iter, n))
                    print("当前总体样本损失函数值为: ", loss)
                    print("当前参数值为: ", self.theta)
                    return self.theta
        print("超过迭代次数，当前总体样本损失函数值为: ", loss)
        print("当前参数值为: ", self.theta)
        return self.theta
    def theoreticalSoving(self):
        '''
        θ=(X^T X)^(−1) X^T
        '''
        X = self.X; t = self.t
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(t)
    # ------------------------------------------------
    def visualization(self, enbale):
        '''
        动态作图，显示图像
        '''
        if not enbale: return
        plt.figure(0); plt.cla()                # 选择figure(0), 并清空画布
        plt.scatter(self.X[:, 1], self.t, c='b')# 描点
        x = np.arange(0, 25, 0.1)
        y = self.theta[0] + self.theta[1] * x   # 直线
        plt.plot(x, y, c='r')
        plt.pause(0.001)
    def drawPrediction(self):
        plt.ioff()
        x = np.arange(self.X.shape[0])
        plt.figure(1); plt.xlim(0, self.X.shape[0])
        plt.scatter(x, self.t, c='b')
        plt.plot(x, self.predict(self.X, self.theta), c='r')
        plt.show()


# --------------------------------------
mdl = LinearRegression(file1, n_batch=3)
"""
n_batch = 1
  第1258次迭代, 第0批数据
  当前总体样本损失函数值为:  8.999986244325596
  当前参数值为:  [-3.39291241  1.14251505]
  理论最优:  [-3.89578088  1.19303364]
  理论最优解下的损失值为:  8.953942751950358
n_batch = -1
  第10次迭代, 第95批数据
  当前总体样本损失函数值为:  8.99311520700548
  当前参数值为:  [-3.43194931  1.14643286]
  理论最优:  [-3.89578088  1.19303364]
  理论最优解下的损失值为:  8.953942751950358
n_batch = 3
  超过迭代次数，当前总体样本损失函数值为:  9.865500020342207
  当前参数值为:  [-4.12352264  1.11059131]
  理论最优:  [-3.89578088  1.19303364]
  理论最优解下的损失值为:  8.953942751950358
"""
# --------------------------------------
# mdl = LinearRegression(file2, n_batch=3)
"""
超过迭代次数，当前总体样本损失函数值为:  nan
当前参数值为:  [nan nan nan]
"""
# --------------------------------------
mdl.fit(min_loss=4.0, learning_rate=0.001, max_iter=10000)
theta_theoretical = mdl.theoreticalSoving()
print('理论最优: ', theta_theoretical)
print('理论最优解下的损失值为: ', mdl.lossFunction(mdl.predict(mdl.X, theta_theoretical), mdl.t))
mdl.visualization(True)
pass    # for debug