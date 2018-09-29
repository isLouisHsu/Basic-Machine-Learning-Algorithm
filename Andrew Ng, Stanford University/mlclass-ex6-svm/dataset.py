import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# ex6data1.mat - Example Dataset 1
file1 = './ex6data1.mat'
# ex6data2.mat - Example Dataset 2
file2 = './ex6data2.mat'
# ex6data3.mat - Example Dataset 3
file3 = './ex6data3.mat'

def load_X_y(file, display=False):
    data = loadmat(file)
    X = data['X'].astype('float'); y = data['y'].reshape((-1,)).astype('int')
    y[y==0] = -1    # 必须为{-1, +1}
    if display:
        plt.figure('data')
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()
    return X, y

if __name__ == '__main__':
    load_X_y(file1, True)
    load_X_y(file2, True)
    load_X_y(file3, True)