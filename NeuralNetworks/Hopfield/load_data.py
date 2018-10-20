import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist():
    '''
    Load mnist data
    http://yann.lecun.com/exdb/mnist/

    60000 training examples
    10000 test sets

    Arguments:
        kind: 'train' or 'test', string charater input with a default value 'train'

    Return:
        xxx_images: n*m array, n is the sample count, m is the feature number which is 28*28
        xxx_labels: class labels for each image, (0-9)
    '''

    root_path = 'D:/machine-learning/datasets/mnist'

    train_labels_path = os.path.join(root_path, 'train-labels.idx1-ubyte')
    train_images_path = os.path.join(root_path, 'train-images.idx3-ubyte')

    test_labels_path = os.path.join(root_path, 't10k-labels.idx1-ubyte')
    test_images_path = os.path.join(root_path, 't10k-images.idx3-ubyte')

    with open(train_labels_path, 'rb') as lpath:
        # '>' denotes bigedian
        # 'I' denotes unsigned char
        magic, n = struct.unpack('>II', lpath.read(8))
        #loaded = np.fromfile(lpath, dtype = np.uint8)
        train_labels = np.fromfile(lpath, dtype = np.uint8).astype(np.float)

    with open(train_images_path, 'rb') as ipath:
        magic, num, rows, cols = struct.unpack('>IIII', ipath.read(16))
        loaded = np.fromfile(train_images_path, dtype = np.uint8)
        # images start from the 16th bytes
        train_images = loaded[16:].reshape(len(train_labels), 784).astype(np.float)

    with open(test_labels_path, 'rb') as lpath:
        # '>' denotes bigedian
        # 'I' denotes unsigned char
        magic, n = struct.unpack('>II', lpath.read(8))
        #loaded = np.fromfile(lpath, dtype = np.uint8)
        test_labels = np.fromfile(lpath, dtype = np.uint8).astype(np.float)

    with open(test_images_path, 'rb') as ipath:
        magic, num, rows, cols = struct.unpack('>IIII', ipath.read(16))
        loaded = np.fromfile(test_images_path, dtype = np.uint8)
        # images start from the 16th bytes
        test_images = loaded[16:].reshape(len(test_labels), 784)    

    return train_images, train_labels, test_images, test_labels
    
def test_mnist_data():
    '''
    Just to check the data

    Argument:
        none

    Return:
        none
    '''
    train_images, train_labels, test_images, test_labels = load_mnist()
    fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True)
    ax =ax.flatten()
    for i in range(10):
        img = train_images[i][:].reshape(28, 28)
        ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')
        print('corresponding labels = %d' %train_labels[i])

if __name__ == '__main__':
    test_mnist_data()