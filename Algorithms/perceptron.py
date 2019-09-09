import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    """ perceptron for classfication

    Attributes:
        lr:         {float}         learing rate
        thresh:     {int}           stop condition
        weights:    {ndarray(n+1)}  weights of perceptron

    Notes:
        - y_pred = \sign(wx + b)
        - w = w + \eta*y_i*x_i

    """

    def __init__(self, lr=0.1, thresh=0.99):
        self.lr = lr
        self.thresh = thresh

        self.weights = None

    def fit(self, X, y):
        """
        Params:
            X:  {ndarray(N, n)}
            y:  {ndarray(N)}    0, 1
        """
        n_sample, n_feature = X.shape
        self.weights = np.random.randn(n_feature + 1)           # initialize weights and bias

        i_iter = 0
        while True:
            ## 计算输出
            y_pred = self.predict(X)
            accuracy = np.mean(y_pred==y)
            print('iter: {:6d} || accuracy: {:.2%}'.format(i_iter, accuracy))
            if accuracy > self.thresh: break
             
            ## 随机选取一个错误分类的样本
            _X = X[y_pred!=y]; _y = y[y_pred!=y]
            _X = np.concatenate([_X, np.ones((_X.shape[0], 1))], axis=1)
            idx = np.random.choice(_X.shape[0], size=1)[0]
            _X = _X[idx]; _y = _y[idx]

            ## 更新权值
            self.weights += self.lr*_X*_y

    def predict(self, X):
        """
        Params:
            X: {ndarray(N, n)}
        Returns:
            y: {ndarray(N)}
        """
        
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) # add x_0
        y_pred = np.sign(X.dot(self.weights))

        return y_pred        


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(centers=2)
    y[y==0] = -1
    plt.figure(0)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    m = Perceptron(lr=0.1)
    m.fit(X, y)
    y_pred = m.predict(X)