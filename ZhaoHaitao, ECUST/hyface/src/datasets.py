import os
import cv2
import numpy as np
import skimage
from scipy.io import loadmat
import matplotlib.pyplot as plt

DEBUG = False

image_path  = 'F:/hyperspectral_face/data'
dirname = ['SampleImages_F', 'SampleImages_L', 'SampleImages_R']
H = 220
W = 180
C = 33

def load_X_y(dirname, mode='train', n_samples=-1, type='F', dsize=None):
    """
    Notes:
        - `SampleImages_{type}_augmented.txt`
        - `SampleImages_{type}_augmented`
            - `{sample}_{index}.npy`
    """
    txtfile = '{}.txt'.format(dirname)
    f = open(os.path.join(image_path, txtfile), 'r')
    filenames = f.readlines()

    if n_samples == -1: n_samples = len(filenames)
    if dsize is None: dsize = (W, H)

    X = np.zeros(shape=(n_samples, dsize[1], dsize[0], C))
    y = np.zeros(shape=(n_samples), dtype='int')
    for i in range(n_samples):
        splitname = filenames[i].strip().split('_')
        if mode == 'train':
            idx_sub = int(splitname[0]); idx_sess = int((splitname[1].split('.'))[0])
        elif mode == 'test':
            idx_sub = int(splitname[2]); idx_sess = int((splitname[3].split('.'))[0])
        X[i] = load_datacube(type, mode, idx_sub, idx_sess, dsize=dsize)
        y[i] = idx_sub
        if DEBUG: show_cube(X[i], pausetime=5)
        if i % 500 == 0:
            print_log = 'sample {}/{}'.format(i+1, n_samples)
            print(print_log)
    f.close()
    return X, y

def load_X_y_with_gray(mode, n_samples=-1, dsize=None):
    if mode == 'train':
        txtfile = 'SampleImages_F_augmented_with_gray.txt'
    elif mode == 'test':
        txtfile = 'SampleImages_F_test.txt'
    
    f = open(os.path.join(image_path, txtfile), 'r')
    filenames = f.readlines()
    
    if n_samples == -1: n_samples = len(filenames)
    if dsize is None: dsize = (W, H)

    X = np.zeros(shape=(n_samples, dsize[1], dsize[0], C + 1))
    y = np.zeros(shape=(n_samples), dtype='int')
    for i in range(n_samples):
        splitname = filenames[i].strip().split('_')
        if mode == 'train':
            idx_sub = int(splitname[0]); idx_sess = int((splitname[1].split('.'))[0])
        elif mode == 'test':
            idx_sub = int(splitname[2]); idx_sess = int((splitname[3].split('.'))[0])
        X[i] = load_datacube_with_gray(mode, idx_sub, idx_sess, dsize)
        y[i] = idx_sub
        if (i + 1) % 500 == 0: print('image {}/{}'.format(i, n_samples))

    return X, y

    # elif mode == 'test':


def augment_datasets(type, n_augment=10):
    """
    Parameters:
        type {str}: 'F'
    Note:
        - 将现有的样本扩充，例如现在样本1有照片5张，先保存原照片，然后在这个照片上作相应镜像、噪声、拉伸、旋转，
        每张为一组，一组n_group(n_augment//5)张，做出(n_group-1)张，这样一共(n_augment)张照片
        - 原数据名保存在`image_path/SampleImages_{type}_train.txt`
        - 将扩充的文件名保存至`image_path/SampleImages_{type}_augmented.txt`
        - 扩充文件保存至`image_path/SampleImages_{type}_augmented`
    """
    txtfile = os.path.join(image_path, 'SampleImages_{}_train.txt'.format(type))
    txtfile_aug = os.path.join(image_path, 'SampleImages_{}_augmented.txt'.format(type))
    dir_aug = os.path.join(image_path, 'SampleImages_{}_augmented'.format(type))
    if not os.path.exists(dir_aug): os.mkdir(dir_aug)

    f = open(txtfile, 'r')

    if not DEBUG:
        f_aug = open(txtfile_aug, 'w')

    dic = {i: [] for i in range(50)}
    filenames = f.readlines()
    for filename in filenames:
        filename = filename.strip()
        splitname = filename.split('_')
        dic[int(splitname[2])].append(filename) # 存储标签对应样本

    for i in range(50):
        srcfiles = dic[i]
        n_samples = len(srcfiles)
        if n_samples == 0: continue	            # 第i个人照片数统计
        n_group = n_augment // n_samples	    # 每张照片扩充一组，计算组数，每组n_group张
        for j in range(n_samples):
            srcfile = srcfiles[j]               # 第j组，用第j张照片扩充
            splitname = srcfile.split('_')
            src = load_datacube(type, None, int(splitname[2]), int(splitname[3][0]))
            if DEBUG: show_cube(src, pausetime=5)
            for k in range(0, n_group):         # 该组的第k张，第一张为原图
                dst = src.copy()
                if k != 0:
                    dst = flip(dst)
                    # dst = noise(dst, gaussian=False, salt=True, seed=None)
                    dst = perspective(dst, 0.1)
                    dst = rotate(dst, 15)
                    if DEBUG: show_cube(dst, pausetime=5)
                print_log = 'sample: {:>2d}/{:>2d}, n_group: {:>2d}/{:>2d}, n_augment: {:>2d}/{:>2d}'.\
                            format(i, 50, j, n_samples, k, n_group)
                print(print_log)
                augfilename = '{}_{}.npy'.format(i, j * n_group + k)
                if not DEBUG:
                    np.save(os.path.join(image_path, 'SampleImages_{}_augmented'.format(type), augfilename), dst)
                    f_aug.write(augfilename + '\n')
    f.close()

    if not DEBUG:
        f_aug.close()

def augment_datasets_with_gray(n_augment=80):
    """
    Parameters:
        n_augment {int}
    Note:
        - 将现有的样本扩充，例如现在样本1有照片5张，先保存原照片，然后在这个照片上作相应镜像、噪声、拉伸、旋转
        - 灰度图也相应变换，最后输出通道为34
    """
    txtfile = os.path.join(image_path, 'SampleImages_F_train.txt')
    txtfile_aug = os.path.join(image_path, 'SampleImages_F_augmented_with_gray.txt')
    dir_aug = os.path.join(image_path, 'SampleImages_F_augmented_with_gray')
    if not os.path.exists(dir_aug): os.mkdir(dir_aug)

    f = open(txtfile, 'r')
    if not DEBUG:
        f_aug = open(txtfile_aug, 'w')

    dic = {i: [] for i in range(50)}
    filenames = f.readlines()
    for filename in filenames:
        filename = filename.strip()
        splitname = filename.split('_')
        dic[int(splitname[2])].append(filename) # 存储标签对应样本

    for i in range(50):
        srcfiles = dic[i]
        n_samples = len(srcfiles)
        if n_samples == 0: continue	            # 第i个人照片数统计
        n_group = n_augment // n_samples	    # 每张照片扩充一组，计算组数，每组n_group张
        for j in range(n_samples):
            srcfile = srcfiles[j]               # 第j组，用第j张照片扩充
            splitname = srcfile.split('_')
            idx_sub, idx_sess = int(splitname[2]), int(splitname[3][0])
            src_hf = load_datacube('F', None, idx_sub, idx_sess)

            srcfile_gray = '{}_{:d}_{:d}.jpg'.format('F', idx_sub, idx_sess)
            src_gray = cv2.imread(os.path.join(image_path, 'SampleImages_F', srcfile_gray), cv2.IMREAD_GRAYSCALE)

            src = np.dstack((src_hf, src_gray))
            if DEBUG: show_cube(src, pausetime=5, with_gray=True)

            for k in range(0, n_group):         # 该组的第k张，第一张为原图
                dst = src.copy()
                if k != 0:
                    dst = flip(dst)
                    # dst = noise(dst, gaussian=False, salt=True, seed=None)
                    dst = perspective(dst, 0.1)
                    dst = rotate(dst, 15)
                    if DEBUG: show_cube(dst, pausetime=5, with_gray=True)
                print_log = 'sample: {:>2d}/{:>2d}, n_group: {:>2d}/{:>2d}, n_augment: {:>2d}/{:>2d}'.\
                            format(i, 50, j, n_samples, k, n_group)
                print(print_log)
                augfilename = '{}_{}.npy'.format(i, j * n_group + k)
                if not DEBUG:
                    np.save(os.path.join(image_path, 'SampleImages_F_augmented_with_gray', augfilename), dst)
                    f_aug.write(augfilename + '\n')

    f.close()
    if not DEBUG:
        f_aug.close()

def load_datacube(type, mode, idx_sub, idx_sess, dsize=None):
    """
    Parameters:
        type {str}: 'F', 'L', or 'R'
        idx_sub {int}: 
        idx_sess {int}: 
        dsize: {tuple(h, w)}
    Notes:
        filename: SampleImages_{type}/HyperFaceCube_{type}_{idx_sub}_{idx_sess}.mat
    """
    if mode == 'train':
        filename = os.path.join(image_path, 'SampleImages_{}_augmented/{:d}_{:d}.npy'.\
                        format(type, idx_sub, idx_sess))
    elif mode == 'test':
        filename = os.path.join(image_path, 'SampleImages_{}_test/HyperFaceCube_{}_{:d}_{:d}.mat'.\
                        format(type, type, idx_sub, idx_sess))
    else:
        filename = os.path.join(image_path, 'SampleImages_{}/HyperFaceCube_{}_{:d}_{:d}.mat'.\
                        format(type, type, idx_sub, idx_sess))

    if not os.path.exists(filename):
        print('file not exists!'); return False

    if mode == 'train':
        datacube = np.load(filename)
    elif mode == 'test':
        datacube = loadmat(filename)['SpecCube_{}'.format(type)]
    else:
        datacube = loadmat(filename)['SpecCube_{}'.format(type)]

    if dsize is not None:
        datacube = resize(datacube, dsize)

    return datacube

def load_datacube_with_gray(mode, idx_sub, idx_sess, dsize=None):
    if mode == 'train':
        filename = os.path.join(image_path, 'SampleImages_F_augmented_with_gray/{:d}_{:d}.npy'.\
                        format(idx_sub, idx_sess))
        if not os.path.exists(filename):
            print('file not exists!'); return False
        datacube = np.load(filename)

    elif mode == 'test':    # 未生成34通道的图，现场组hhhh
        imgpath  = os.path.join(image_path, 'SampleImages_F_test')

        cubename = os.path.join(imgpath, 'HyperFaceCube_F_{}_{}.mat'.format(idx_sub, idx_sess))
        if not os.path.exists(cubename):
            print('file not exists!'); return False
        datacube = loadmat(cubename)['SpecCube_F']

        grayname = os.path.join(imgpath, 'F_{}_{}.jpg'.format(idx_sub, idx_sess))
        if not os.path.exists(grayname):
            print('file not exists!'); return False
        datagray = cv2.imread(grayname, cv2.IMREAD_GRAYSCALE)

        datacube = np.dstack((datacube, datagray))

    if dsize is not None:
        datacube = resize(datacube, dsize)
    
    return datacube

def show_cube(datacube, pausetime=5, with_gray=False):
    """ show a sample
    Parameters:
        datacube {ndarray(H, W, C)}
    """
    (h, w) = datacube.shape[:2]; dtype = datacube.dtype
    imgshow = np.zeros(shape=(3*h, 11*w), dtype=dtype)
    for i in range(33):
        channel = datacube[:, :, i]
        idxH = i // 11; idxW = i % 11
        imgshow[idxH*h:(idxH+1)*h, idxW*w:(idxW+1)*w] = channel
    
    if DEBUG: 
        cv2.imshow('datacube', imgshow.astype('uint8'))   # 显示有问题
        if with_gray: cv2.imshow('gray', datacube[:,:,-1].astype('uint8'))
        cv2.waitKey(pausetime)
    else:
        plt.figure('datacube')
        plt.imshow(imgshow)
        if with_gray: 
            plt.figure('gray')
            plt.imshow(datacube[:,:,-1])
        plt.show()



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
# def saltnoise(image, salt=0.0):
#     """ add salt & pepper and gaussian noise
#     Parameters:
#         image {ndarray(H, W, C)}
#         salt {float(0, 1)} number of salt pixel = salt*h*w
#     Notes:
#         TODO: gaussain noise
#     """
#     (h, w) = image.shape[:2]
#     n_salt = int(salt * h * w)
#     for n in range(n_salt):
#         hr = np.random.randint(0, h)
#         wr = np.random.randint(0, w)
#         issalt = (np.random.rand(1) > 0.5)
#         image[hr, wr] = 255 if issalt else 0
#     return image
# def gaussiannoise(image, mu, sigma, k): 
#     pass
def noise(image, gaussian, salt, seed=None):
    """ add noise to image TODO
    Parameters:
        image {ndarray(H, W, C)}
        gaussian {bool}: 
        salt {bool}: 
    """
    dtype = image.dtype
    if gaussian:
        image = skimage.util.random_noise(image, mode='gaussian', seed=seed)
    if salt:
        image = skimage.util.random_noise(image, mode='s&p', seed=seed)

    image = (image * 255).astype(dtype)
    return image
def brightcontrast(image, brtadj=0, cstadj=1.0):
    """ adjust bright and contrast value
    Parameters:
        image {ndarray(H, W, C)}
        brtadj {int}    adjust bright
        cstadj {float}  adjust contrast
    """
    # dtype = image.dtype
    # image = image.astype('int32')*cstadj + brtadj
    # image = np.clip(image, 0, 255).astype(dtype)
    # return image
    dtype = image.dtype
    image = image.astype('int32')*cstadj + brtadj
    image = np.clip(image, 0, 0xFFFF).astype(dtype)
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


# def load_X_y(type, mode=None, dsize=None):
#     """
#     Parameters:
#         type {str}: 'F', 'L', or 'R'
#         mode {str}: 'train', 'test'
#     Notes:
#         从文本'image_path/SampleImages_{type}.txt'导入原有的`.mat`数据，'image_path/SampleImages_{type}'
#     """
#     if mode is None:
#         txtfile = os.path.join(image_path, 'SampleImages_{}.txt'.format(type))
#     else:
#         txtfile = os.path.join(image_path, 'SampleImages_{:>4d}_{}.txt'.format(type, mode))
#     if not os.path.exists(txtfile): print('file not exists!'); return
    
#     f = open(txtfile, 'r')

#     filenames = f.readlines()
#     n_samples = len(filenames)
#     X = np.zeros(shape=(n_samples, H, W, C))
#     y = np.zeros(shape=(n_samples), dtype='int')
#     for i in range(n_samples):
#         splitname = filenames[i].strip().split('_')
#         idx_sub = int(splitname[2]); idx_sess = int(splitname[3][0])
#         X[i] = load_datacube(type, False, idx_sub, idx_sess, dsize)
#         y[i] = idx_sub

#     f.close()
#     return X, y

# def read_filename(dirname):
#     """ 读取文件夹中的文件名，保存到`.txt`
#     Parameters:
#         dirname {str}: name of directory
#     """
#     dirpath = os.path.join(image_path, dirname)
#     filenames = os.listdir(dirpath)
#     with open(os.path.join(image_path, '{}.txt'.format(dirname)), 'w') as f:
#         for file in filenames:
#             if file[:-4:-1][::-1] == 'mat':
#                 f.write(file + '\n')

# def merge_datasets():
#     """ 将三角度拍摄的照片合并，文件名按sample序列保存到`SampleImages_merged.txt`
#     """
#     dic = {i: [] for i in range(50)}
#     for dir in dirname:
#         with open(os.path.join(image_path, '{}.txt'.format(dir)), 'r') as f:
#             filename = f.readline().strip()
#             while filename:
#                 dic[int(filename.split('_')[2])].append(filename)
#                 filename = f.readline().strip()

#     f = open(os.path.join(image_path, 'SampleImages_merged.txt'), 'w')
#     for i in range(50):
#         for filename in dic[i]:
#             f.write(filename + '\n')
#     f.close()








if __name__ == '__main__':
    # load_datacube('R', False, 1, 2)
    # read_filename()
    # merge_datasets()
    # X, y = load_X_y('SampleImages_F_test', mode='test', n_samples=-1, type='F', dsize=None)
    # X, y = load_X_y('SampleImages_F_augmented', mode='train', n_samples=800, dsize=(18, 22))

    # img = cv2.imread('F:/hyperspectral_face/data/SampleImages_F/F_1_1.jpg')
    img = load_datacube_with_gray('train', 1, 1)[:, :, 17]
    # img = noise(img, True, True)
    # img = resize(img, (110, 90))
    # img = flip(img)
    # img = rotate(img, 10)
    # img = noise(img, 0.007)
    # img = brightcontrast(img, 50000)
    # img = perspective(img, d=50)
    # cv2.imshow('img', img); cv2.waitKey(0)
    plt.imshow(img); plt.show()

    # augment_datasets('F', n_augment=80)
    # X, y = load_X_y('SampleImages_F_augmented', mode='train', n_samples=800, dsize=(18, 22))

    # augment_datasets_with_gray(n_augment=80)
    # X = load_datacube_with_gray('train', 1, 1, dsize=(18, 22))
    # X = load_datacube_with_gray('test', 2, 6, dsize=(18, 22))
    # X, y = load_X_y_with_gray('test', n_samples=-1, dsize=(18, 22))
    # show_cube(X[0], with_gray=True)
    pass
