from PCA import PrincipalComponentAnalysis
from load_data import load_2D, facefile
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import MinMaxScaler

n_face = 100
W = H = 32                          # 降维前图像大小
W_reduced = H_reduced = 16          # 降维后的图像大小

X = load_2D(facefile)[: n_face]     # 前100张脸的图像数据 - 100x1024 - pixel∈(-128 ~ 127)
X = MinMaxScaler(feature_range=(0, 255)).fit_transform(X).astype('uint8')

reduce_dim = PrincipalComponentAnalysis(n_component = W_reduced*H_reduced)
X_reduced = reduce_dim.fit_transform(X)
X_restructed = reduce_dim.transform_inv(X_reduced)

# ---------------- 作图显示 ----------------
def display(X, filename, dsize=(48, 48)):
    n_H_img = n_W_img = int(np.sqrt(X.shape[0]))        # 每张脸的索引
    H_i = W_i = int(np.sqrt(X.shape[1]))                # 每张脸上像素的索引
    showImg = np.zeros(shape=(n_H_img*dsize[0], n_W_img*dsize[1]), dtype=X.dtype)
    for h in range(n_H_img):
        for w in range(n_W_img):
            X_i = X[w*n_W_img + h]                      # 获取第i张脸的数据，i = w*n_W_img + h
            X_i_reshaped = X_i.reshape((W_i, H_i))      # 展开成二维
            w_n1, w_n2 = w*dsize[0], (w+1)*dsize[0]
            h_n1, h_n2 = h*dsize[1], (h+1)*dsize[1]
            X_i_resized = cv2.resize(X_i_reshaped, dsize=dsize)  # 图像缩放到dsize
            showImg[w_n1: w_n2, h_n1: h_n2] = X_i_resized.T   # 保存到图片上
    cv2.imwrite(filename, showImg)


X = X.astype('uint8')
display(X, filename='./data/face_origin.png')

X_reduced = MinMaxScaler(feature_range=(0, 255)).fit_transform(X_reduced.astype('float64'))
display(X_reduced, filename='./data/face_reduced.png')

X_restructed = MinMaxScaler(feature_range=(0, 255)).fit_transform(X_restructed.astype('float64'))
display(X_restructed, filename='./data/face_restructed.png')

