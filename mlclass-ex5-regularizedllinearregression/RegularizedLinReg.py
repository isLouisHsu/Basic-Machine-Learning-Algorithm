import numpy as np
import matplotlib.pyplot as plt
import load_data
from sklearn.cross_validation import train_test_split

class RegularizedLinearRegression():
    def __init__(self, n_ploy=1, n_batch=-1, regularize=0.0):
        self.n_ploy = n_ploy
        self.n_batch = n_batch
        self.regularize = regularize
        self.theta = None
    def fit(self, X, y, learning_rate=0.01, max_iter=5000, min_loss=10):
        # --------------- 数据预处理部分 ---------------
        # 加入全1列
        X = np.c_[np.ones(shape=(X.shape[0])), X]
        # 构造高次特征
        if self.n_ploy > 1:
            for i in range(2, self.n_ploy + 1):
                X = np.c_[X, X[:, 1]**i]
        # ---------------- 参数迭代部分 ----------------
        # 初始化参数
        self.theta = np.random.uniform(-1, 1, size=(X.shape[1],))
        # 数据批次
        n_batch = X.shape[0] if self.n_batch==-1 else self.n_batch
        batch_size = X.shape[0] // n_batch
        # 停止条件
        n_iter = 0; loss = float('inf')
        # 开始迭代
        while n_iter < max_iter:
            n_iter += 1
            for n in range(n_batch):
                n1, n2 = n*batch_size, (n+1)*batch_size
                X_batch = X[n1: n2]; y_batch = y[n1: n2]
                ####
                grad = self.lossFunctionDerivative(X_batch, y_batch)
                self.theta -= learning_rate * grad
                ####
                loss = self.score(y_batch, self.predict(X_batch))
                if loss < min_loss:
                    print('第%d次迭代, 第%d批数据' % (n_iter, n))
                    print("当前总体样本损失为: ", loss)
                    return self.theta
            if n_iter%100 == 0:
                print('第%d次迭代' % n_iter)
                print("当前总体样本损失为: ", loss)
        print("超过迭代次数")
        print("当前总体样本损失为: ", loss)
        return self.theta

    def predict(self, X):
        '''
        预测线性回归结果
        '''
        return X.dot(self.theta)
    def lossFunctionDerivative(self, X, y):
        '''
        损失函数的梯度，损失函数采用mse
        '''
        y_pred = self.predict(X)
        # theta = self.theta;     # ！注意：theta = self.theta 不仅仅是赋值，类似引用，修改theta会影响self.theta
        theta = self.theta.copy()
        theta[0] = 0            # θ0不需要正则化
        return (X.T.dot(y_pred - y) + self.regularize * theta) / X.shape[0]
    def score(self, y_true, y_pred):
        '''
        预测评分，采用mse，即lossFunction
        '''
        err = y_pred - y_true
        return 0.5 * np.mean(np.square(err))

if __name__ == '__main__':
    X = np.arange(-5, 5, 1)
    y = 10 * X**2 + 20 * X + 30
    y = y.astype('float')
    y += np.random.rand(y.shape[0])

    estimator1 = RegularizedLinearRegression(n_ploy=2, n_batch=1, regularize=0.0)
    estimator1.fit(X, y, learning_rate=0.01, max_iter=5000, min_loss=0)

    estimator2 = RegularizedLinearRegression(n_ploy=2, n_batch=1, regularize=10.0)
    estimator2.fit(X, y, learning_rate=0.01, max_iter=5000, min_loss=0)
    
    print(estimator1.theta)
    print(estimator2.theta)

    '''
    [30.49931186 19.97259836  9.98883834]
    [33.51393343 17.3600116   9.48049541]
    '''