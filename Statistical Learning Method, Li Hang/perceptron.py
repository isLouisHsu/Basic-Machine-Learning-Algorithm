import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    '''
    @note:  the loss function of preceptron is
                loss = - Σyi(w*xi+b) = - y^T(xw)
                y - (n_samples,)
                w - (n_features + 1,)
                x - (m_samples, n_features + 1)
            so the gradient is
                dloss = -X.T(y)
    '''
    def __init__(self):
        self.theta = None
    def fit(self, X, y, learning_rate=0.01, max_iter=5000, min_acc=0.9): 
        plt.ion(); X_figure = np.linspace(0, 5)
        # 加上全1列
        X_1 = np.c_[np.ones(shape=(X.shape[0],)), X]
        self.theta = np.random.rand(X_1.shape[0])
        # 迭代
        n_iter = 0; acc = 0
        while n_iter < max_iter:
            n_iter += 1
            # ---
            # grad = - X_1.T.dot(y)                                 # 全部样本用于更新参数
            # self.theta -= learning_rate * grad
            # ---
            for i in range(X_1.shape[0]):
                if y[i] * self.predict(X[i].reshape((1, -1))) < 0:  # 选取分类错误的样本点用于更新参数
                    grad = - y[i] * X_1[i]                          # 注意符号
                    self.theta -= learning_rate * grad
            # ---
            plt.figure(0); plt.cla()
            plt.scatter(X[:, 0], X[:, 1], c=y)
            y_figure = - (self.theta[0] + self.theta[1] * X_figure) / self.theta[2]
            plt.plot(X_figure, y_figure)
            plt.pause(0.2)
            acc = self.score(y, self.predict(X))
            if acc > min_acc:
                print('迭代结束，共迭代%d次， 当前准确率%f' % (n_iter, acc))
                break
        if n_iter >= max_iter:
            print('超过迭代次数，当前准确率%f' % acc)
        pass
    def predict_val(self, X):
        # 加上全1列
        X_1 = np.c_[np.ones(shape=(X.shape[0],)), X]
        return X_1.dot(self.theta)
    def predict(self, X):
        return np.sign(self.predict_val(X))
    def score(self, y_true, y_pred):
        ''' accuracy '''
        return np.mean(np.equal(y_true, y_pred).astype('float'))

if __name__ == '__main__':
    X = np.array([
        [3, 3], [4, 3], [1, 1]
    ])
    y = np.array([
        1, 1, -1
    ])

    estimator = Perceptron()
    estimator.fit(X, y, learning_rate=0.1)
    y_pred = estimator.predict(X)
    print(estimator.score(y, y_pred))