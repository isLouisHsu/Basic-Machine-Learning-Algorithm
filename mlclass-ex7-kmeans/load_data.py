import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat

# Example Dataset for PCA
data_pca = './data/ex7data1.mat'
# Example Dataset for K-means
data_kmeans = './data/ex7data2.mat'
# Faces Dataset
facefile  = './data/ex7faces.mat'
# Example Image
imagefile = './data/bird_small.png'

def load_2D(filename, display=False):
    data = loadmat(filename)
    X = data['X']
    if display:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()
    return X

def load_image(display=False):
    img = cv2.imread(imagefile)
    if display: 
        # plt.imshow(img)
        # plt.show()
        cv2.imshow('img', img)
        cv2.waitKey(2)
        # cv2.destroyAllWindows()
    return np.array(img)

if __name__ == '__main__':
    # load_2D(data_kmeans, display=True)
    img = load_image(display=True)
    pass
