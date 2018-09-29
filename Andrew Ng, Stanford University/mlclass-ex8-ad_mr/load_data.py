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
# - num movies × num users
# 'R' - (1682, 943) - an binary-valued indicator matrix, 
#                       where R(i; j) = 1 if user j gave a rating to movie i, 
#                       and R(i; j) = 0 otherwise. 
# 'Y' - (1682, 943) - stores the ratings y(i;j)(from 1 to 5)
MRfile = './data/ex8_movies.mat'

# - Parameters provided for debugging
DGBfile = './data/ex8_movieParams.mat'

def load_X_X_yval(filename):
    data = loadmat(filename)
    X = data['X']; Xval = data['Xval']; yval = data['yval'].reshape((-1, ))
    return X, Xval, yval

def load_movie_review_R_Y():
    data = loadmat(MRfile)
    return data['R'].astype('bool'), data['Y'].astype('float')

if __name__ == '__main__':
    load_movie_review()
    pass
