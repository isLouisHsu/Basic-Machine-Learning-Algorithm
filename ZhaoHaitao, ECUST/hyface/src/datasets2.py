import os
import cv2
import time
import numpy as np

import skimage
from scipy.io import loadmat
import matplotlib.pyplot as plt

DEBUG = True

(H, W, C) = (220, 180, 33)

workpath = 'F:/hyperspectral_face/'
sourcedir = 'SampleImages_F'
traindir = 'SampleImages_F_train'
testdir = 'SampleImages_F_test'

def get_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())

def load_train(dsize=None):
    """ 读取训练集数据
    """
    datadir = traindir
    n_subjects = 25; n_sessions = 10 * 3
    n_samples = n_subjects * n_sessions	# 1500

    datapath = os.path.join(workpath, datadir)
    if not os.path.exists(datapath): print("directory doesn't exist!"); return

    if dsize is not None: (w, h) = dsize
    else: (w, h) = (W, H)

    X_train = np.zeros(shape=(n_samples, h, w, C + 1))
    y_train = np.zeros(shape=(n_samples))
    for i_subjects in range(n_subjects):
        for i_sessions in range(n_sessions):
            dataname = os.path.join(
                datapath, "F_{:d}_{:d}.npy".\
                            format(i_subjects + 1, i_sessions))
            i_samples = i_subjects * 20 + i_sessions
            data = np.load(dataname)
            if dsize is not None: data = resize(data, dsize)
            X_train[i_samples] = data
            y_train[i_samples] = i_subjects
        print_log = get_time() + ' subject {} loaded!'.format(i_subjects + 1)
        print(print_log)
    return X_train, y_train

def load_test(dsize=None):
    """ 读取测试集数据
    """
    datadir = testdir
    n_subjects = 25; n_samples = n_subjects

    datapath = os.path.join(workpath, datadir)
    if not os.path.exists(datapath): print("directory doesn't exist!"); return

    if dsize is not None: (w, h) = dsize
    else: (w, h) = (W, H)

    X_test = np.zeros(shape=(n_samples, h, w, C + 1))
    y_test = np.zeros(shape=(n_samples))
    for i_subjects in range(n_subjects):
        dataname = os.path.join(
                datapath, "F_{:d}.npy".\
                        format(i_subjects + 1))
        data = np.load(dataname)
        if dsize is not None: data = resize(data, dsize)
        X_test[i_subjects] = data
        y_test[i_subjects] = i_subjects
        print_log = get_time() + ' subject {} loaded!'.format(i_subjects + 1)
        print(print_log)
    return X_test, y_test

def make_trainset():
    """ 将1-25的样本前3个样本扩增
    Notes:
        - 高光谱数据: `HyperFaceCube_F_<subIdx>_<sessIdx>.mat`
        - 灰度图数据: `F_<subIdx>_<sessIdx>.jpg`
        - 扩增后数据: `F_<subIdx>_<sessIdx>.npy`
        - 归一化至(0, 1)范围内
    """
    srcdir = sourcedir
    dstdir = traindir
    n_subjects = 25; n_sessions = 3; n_augments = 10
    prop = 0.05; degree = 10

    srcpath = os.path.join(workpath, srcdir)
    dstpath = os.path.join(workpath, dstdir)
    if not os.path.exists(srcpath): print("source directory doesn't exist!"); return
    if not os.path.exists(dstpath): os.mkdir(dstpath)

    for i_subjects in range(n_subjects):
        for i_sessions in range(n_sessions):
            matpath = os.path.join(
                        srcpath, 'HyperFaceCube_F_{:d}_{:d}.mat'.format(i_subjects + 1, i_sessions + 1))
            matdata = loadmat(matpath)['SpecCube_F']
            jpgpath = os.path.join(
                        srcpath, 'F_{:d}_{:d}.jpg'.format(i_subjects + 1, i_sessions + 1))
            jpgdata = cv2.imread(jpgpath, cv2.IMREAD_GRAYSCALE)
            matdata = matdata / 0xffff; jpgdata = jpgdata / 0xff
            data = np.dstack((jpgdata, matdata))
            for i_augments in range(n_augments):
                data_i = data.copy()
                dataname = "F_{:d}_{:d}.npy".\
                        format(i_subjects + 1, i_sessions * n_augments + i_augments)
                data_i = flip(data_i)               # 以0.5几率翻转
                data_i = perspective(data_i, prop)  # 以一定范围拉伸
                data_i = rotate(data_i, degree)     # 以±10°旋转
                if DEBUG: plt.figure('datacube'); plt.imshow(data_i[:, :, 0]); plt.show()
                else: np.save(os.path.join(dstpath, dataname), data_i)
            print_log = get_time() + ' subject {}, session {} finished!'.format(i_subjects + 1, i_sessions + 1)
            print(print_log)

def make_testset():
    """ 将1-25的样本第4个样本保存为测试集
    Parameters:
    Notes:
        - 高光谱数据: `HyperFaceCube_F_<subIdx>_<sessIdx>.mat`
        - 灰度图数据: `F_<subIdx>_<sessIdx>.jpg`
        - 保存为数据: `F_<subIdx>_<sessIdx>.npy`
        - 归一化至(0, 1)范围内
    """
    srcdir = sourcedir
    dstdir = testdir
    n_subjects = 25; i_sessions = 1
    
    srcpath = os.path.join(workpath, srcdir)
    dstpath = os.path.join(workpath, dstdir)
    if not os.path.exists(srcpath): print("source directory doesn't exist!"); return
    if not os.path.exists(dstpath): os.mkdir(dstpath)

    for i_subjects in range(n_subjects):
        dataname = "F_{:d}.npy".\
                        format(i_subjects + 1)
        matpath = os.path.join(
                        srcpath, 'HyperFaceCube_F_{:d}_{:d}.mat'.format(i_subjects + 1, i_sessions + 1))
        matdata = loadmat(matpath)['SpecCube_F']
        jpgpath = os.path.join(
                        srcpath, 'F_{:d}_{:d}.jpg'.format(i_subjects + 1, i_sessions + 1))
        jpgdata = cv2.imread(jpgpath, cv2.IMREAD_GRAYSCALE)
        matdata = matdata / 0xffff; jpgdata = jpgdata / 0xff
        data = np.dstack((jpgdata, matdata))
        if DEBUG: plt.figure('datacube'); plt.imshow(data[:, :, 0]); plt.show()
        else: np.save(os.path.join(dstpath, dataname), data)
        print_log = get_time() + ' subject {}, session {} finished!'.format(i_subjects + 1, i_sessions + 1)
        print(print_log)

def show_cube(datacube):
    """ 显示一组样本的各个通道数据
    """
    (h, w) = datacube.shape[:2]; dtype = datacube.dtype
    imgshow = np.zeros(shape=(6*h, 6*w), dtype=dtype)
    for i in range(C + 1):
        channel = datacube[:, :, i]
        idxH = i // 6; idxW = i % 6
        imgshow[idxH*h:(idxH+1)*h, idxW*w:(idxW+1)*w] = channel
    plt.figure('datacube')
    plt.imshow(imgshow)
    plt.show()


def read_filename(dirname, type=None):
    """ 读取文件夹中的文件名，保存在该目录下`<dirname>.txt`中
    Parameters:
        dirname {str}: name of directory
        type {str}: 'mat', 'jpg', 'npy'
    """
    dirpath = os.path.join(workpath, dirname)
    if not os.path.exists(dirpath): print("directory doesn't exist!"); return
    
    if type is not None:
        txtpath = os.path.join(workpath, dirname, '{}_{}.txt'.format(dirname, type))
    else:
        txtpath = os.path.join(workpath, dirname, '{}.txt'.format(dirname))
    filenames = os.listdir(dirpath)

    if not DEBUG:
        f = open(txtpath, 'w')
        for filename in filenames:
            if type is None: 
                f.write(filename + '\n')
            elif filename.split('.')[1] == type:
                f.write(filename + '\n')
        f.close()

def resize(image, dsize):
    """
    Parameters:
        image {ndarray(H, W, C)}
        dsize {tuple(w, h)}
    Notes:
        注意会调换h和w
    """
    return cv2.resize(image, dsize)
def flip(image):
    """
    Parameters:
        image {ndarray(H, W, C)}
    """
    rand_var = np.random.random()
    image = image[:, ::-1, :] if rand_var > 0.5 else image
    return image
def rotate(image, degree):
    """
    Parameters:
        image {ndarray(H, W, C)}
        degree {float}
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    random_angel = np.random.randint(-degree, degree)
    M = cv2.getRotationMatrix2D(center, random_angel, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    return image
def perspective(image, prop):
    """ 透射变换
    Parameters:
        image {ndarray(H, W, C)}
        prop {float}: 在四个顶点多大的方格内选取新顶点，方格大小为(H*prop, W*prop)
    Notes:
        在四个顶点周围随机选取新的点进行仿射变换，四个点对应左上、右上、左下、右下
    """
    (h, w) = image.shape[:2]

    ptsrc = np.zeros(shape=(4, 2))
    ptdst = np.array([[0, 0], [0, w], [h, 0], [h, w]])
    for i in range(4):
        hr = np.random.randint(0, int(h*prop))
        wr = np.random.randint(0, int(w*prop))
        if i == 0:
            ptsrc[i] = np.array([hr, wr])
        elif i == 1:
            ptsrc[i] = np.array([hr, w - wr])
        elif i == 2:
            ptsrc[i] = np.array([h - hr, wr])
        elif i == 3:
            ptsrc[i] = np.array([h - hr, w - wr])
    M = cv2.getPerspectiveTransform(ptsrc.astype('float32'), ptdst.astype('float32'))
    image = cv2.warpPerspective(image, M, (w, h))
    return image
def brightcontrast(image, brtadj=0, cstadj=1.0):
    """ adjust bright and contrast value
    Parameters:
        image {ndarray(H, W, C)} dtype =
        brtadj {int}    adjust bright
        cstadj {float}  adjust contrast
    """
    # dtype = image.dtype
    # image = image.astype('int32')*cstadj + brtadj
    # image = np.clip(image, 0, 0xFFFF).astype(dtype)
    # return image
    image = image*cstadj + brtadj
    image = np.clip(image, 0, 1.0)
    return image

if __name__ == '__main__':
    # read_filename("SampleImages_F", 'mat')
    # read_filename("SampleImages_F", 'jpg')
    make_trainset()
    # read_filename("SampleImages_F_train")
    make_testset()
    # read_filename("SampleImages_F_test")
    # X_train, y_train = load_train()
    # X_test,  y_test  = load_test()
    pass