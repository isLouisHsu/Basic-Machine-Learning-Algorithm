import numpy as np
from scipy.io import loadmat

# X: handwriting images in the shape of 5000 x 400
# y: labels of the images in the shape if 5000 x 1
dataFile = './ex3data1.mat'
# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
weightFile = './ex3weights.mat'

def load_X_y(one_hot=False):
    """
    载入手写数据集，返回NumPy数组
    存在的问题：图片为0~9，标签为1~10，不清楚10指0还是9，下面认作9处理，对结果无影响
    """
    X_y = loadmat(dataFile)
    X = X_y['X']; y = X_y['y'] - 1
    # one-hot 编码
    if one_hot==True:
        n_label = len(set(list(y.reshape((-1,)))))
        encode = np.zeros(shape=(y.shape[0], n_label))
        for i in range(y.shape[0]):
            encode[i, y[i]] = 1
        y = encode
    return X, y

def load_weight():
    theta = loadmat(weightFile)
    theta1 = theta['Theta1']  # 401 x 25, theta1[:, 0]为bias1
    weight1 = theta1[:, 1:].T; bias1 = theta1[:, 0]
    theta2 = theta['Theta2']  # 26 x 10, theta2[:, 0]为bias2
    weight2 = theta2[:, 1:].T; bias2 = theta2[:, 0]
    return weight1, weight2, bias1, bias2

if __name__ == '__main__':
    X, y = load_X_y(one_hot=True)
    weight1, weight2, bias1, bias2 = load_weight()