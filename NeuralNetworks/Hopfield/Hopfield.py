import matplotlib.pyplot as plt
import numpy as np
from load_data import load_mnist

sigmoid = lambda x: 1 / (1 + np.e**(-x))

class Hopfield_MNIST():
    def __init__(self):
        self.W = None
    def fit(self, X, y):
        """ 训练权值 W[i] = \frac{1}{N} \sum_i^N X_i^TX_i
        """
        self.W = np.zeros((10, X.shape[1], X.shape[1]))
        for i in range(10):
            X_i = X[y==i]
            self.W[i] += X_i.T.dot(X_i)
        self.W /= X.shape[0]
        return self.W

    def predict(self, X):
        plt.figure('orgin')
        plt.imshow((X * 255).astype('uint8').reshape((28, 28)))

        X_res = np.zeros(shape=(10, 28, 28))
        for i in range(10):
            X_res[i] = sigmoid(self.W[i].dot(X)).reshape((28, 28))
            X_res[i] = (X_res[i] * 255).astype('uint8')

        X_fig = np.zeros(shape=(2*28, 5*28), dtype='uint8')
        for i in range(2):
            for j in range(5):
                X_res_ = X_res[i * 5 + j]
                X_fig[i*28: (i+1)*28, j*28: (j+1)*28] = X_res[i * 5 + j]

        plt.figure('res')
        plt.imshow(X_fig)
        plt.show()

        pass

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_mnist()
    train_images[train_images>0] = 1

    estimator = Hopfield_MNIST()
    estimator.fit(X=train_images, y=train_labels)
    test_labels_pred = estimator.predict(X=train_images[8])
    pass
