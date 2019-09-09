import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

class LinearRegression():
    """
    Attributes:
        w: {ndarray(n_features + 1, )}
    """
    def __init__(self):
        self.w = None
    def fit(self, X, y, lr=1e-2, epsilon=1e-3, max_iter=5000):
        """ 训练模型
        Args:
            X: {ndarray(n_samples, n_features)}
            y: {ndarray(n_samples,)}
            lr: {float}, learning rate
        """
        X = np.c_[np.ones(X.shape[0]), X]; n_features = X.shape[1]
        self.w = np.random.normal(loc=0, scale=0.1, size=(n_features,)) # initialize
        _loss = float('inf')

        for n_iter in range(max_iter):
            a = self.grad(X, y)
            self.w -= lr * self.grad(X, y)
            y_pred = self.predict(X, True)
            _loss = self.loss(y, y_pred)
            print("loss: {:2.2f}".format(_loss))
            if _loss < epsilon: break
        return self.w
    def predict(self, X, istrain=False):
        """ 预测
        Args:
            X: {ndarray(n_samples, n_features)}
        """
        if not istrain:
            X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self.w)
    def grad(self, X, y):
        """ 计算梯度
        Args:
            X: {ndarray(n_samples, n_features)}
            y: {ndarray(n_samples,)}
        """
        y_pred = self.predict(X, True)
        return X.T.dot(y_pred - y)
    def loss(self, y_true, y_pred):
        """ 计算损失
        Args:
            y_true: {ndarray(n_samples,)}
            y_pred: {ndarray(n_samples,)}
        Notes:
            loss = \frac{1}{2N} \sum_{i=1}^N ({y_true}_i - {y_pred}_i)^2
        """
        err = y_pred - y_true
        return 0.5 * np.mean(np.linalg.norm(err))
    def theoretical_solving(self, X, y, c=1e-6):
        """ 最优解
        Args:
            X: {ndarray(n_samples, n_features)}
            y: {ndarray(n_samples,)}
            c: {float} 
        Notes:
            w = (X^T X + \lambda I)^{(-1)} X^T y
        """
        X = np.c_[np.ones(X.shape[0]), X]
        XTX = X.T.dot(X) + c * np.eye(X.shape[1])
        pinv = np.linalg.inv(XTX).dot(X.T)
        return pinv.dot(y)
    def showfig(self, X, y, w=None):
        """ show figure
        Args:
            X: {ndarray(n_samples, n_features)}
            y: {ndarray(n_samples,)}
            line: {bool} if true, draw the regression model
        Notes:
            - Only for 1-dim samples;
            - y = w[0] + w[1] * x
        """
        plt.figure()
        plt.scatter(X, y, c='b')
        if w is not None:
            X_ = np.linspace(np.min(X), np.max(X))
            y_ = w[0] + w[1] * X_
            plt.plot(X_, y_, c='r')
        plt.show()

if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=1, bias=10., noise=10.0)

    estimator = LinearRegression()
    estimator.showfig(X, y)         # 原始数据显示

    w1 = estimator.fit(X, y, max_iter=500)
    w2 = estimator.theoretical_solving(X, y)

    estimator.showfig(X, y, w1)     # show fig1
    estimator.showfig(X, y, w2)     # show fig2

            