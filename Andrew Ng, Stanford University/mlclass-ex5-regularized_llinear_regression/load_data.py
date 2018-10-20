import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Dataset
dataFile = './ex5data1.mat'

def load_X_y(display=False):
    """
    载入手写数据集，返回NumPy数组
    存在的问题：图片为0~9，标签为1~10，不清楚10指0还是9，下面认作9处理，对结果无影响
    """
    data = loadmat(dataFile)
    X = data['X']; y = data['y'].reshape(-1)
    if display:
        plt.figure('data')
        plt.scatter(X, y)
        plt.show()
    return X, y

if __name__ == '__main__':
    X, y = load_X_y(True)