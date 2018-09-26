import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# - First example Dataset for anomaly detection
# 'X'    - (307, 2) 
# 'Xval' - (307, 2)
# 'yval' - (307, 1)
ADfile1 = './data/ex8data1.mat'

# - Second example Dataset for anomaly detection
# 'X'    - (1000, 11)
# 'Xval' - ( 100, 11)
# 'yval' - ( 100,  1)
ADfile2 = './data/ex8data2.mat'

# 'X' is used to estimate the Gaussian distribution
# 'Xval', 'yval' is used to select the threshold з
# ----------------------------------------------------
# - Movie Review Dataset
# 'R' - (1682, 943)
# 'Y' - (1682, 943)
MRfile = './data/ex8_movies.mat'

# - Parameters provided for debugging
DGBfile = './data/ex8_movieParams.mat'

def load_X_X_yval(filename):
    data = loadmat(filename)
    X = data['X']; Xval = data['Xval']; yval = data['yval'].reshape((-1, ))
    # if s
    return X, Xval, yval


if __name__ == '__main__':
    # data = load(ADfile1)
    load_X_X_yval(ADfile1)
    pass
