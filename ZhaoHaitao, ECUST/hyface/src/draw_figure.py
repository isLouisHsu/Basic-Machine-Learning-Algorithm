import os
import numpy as np
import matplotlib.pyplot as plt

workpath = 'F:/hyperspectral_face/'
resultdir = 'results'

def main():
    acc_train  = np.load(os.path.join(workpath, resultdir, 'exp1_acc_train.npy'))[1:]
    acc_test   = np.load(os.path.join(workpath, resultdir, 'exp1_acc_test.npy'))[1:]
    loss_train = np.load(os.path.join(workpath, resultdir, 'exp1_loss_train.npy'))[1:]
    loss_test = np.load(os.path.join(workpath, resultdir, 'exp1_loss_test.npy'))[1:]

    channels = np.arange(33) + 1

    plt.figure(figsize=(10, 4))
    plt.title('single channel(pca&svm)'); plt.xlabel('channels'); plt.ylabel('accuracy/%')
    plt.bar(
        channels, acc_train * 100, alpha=0.9, width=0.35, facecolor='lightblue', edgecolor='white', label='train'
    )
    plt.bar(
        channels + 0.35 , acc_test * 100,  alpha=0.9, width=0.35, facecolor='yellowgreen', edgecolor='white', label='valid'
    )
    plt.legend(loc='upper right')

    plt.figure(figsize=(10, 4))
    plt.title('single channel(pca&svm)'); plt.xlabel('channels'); plt.ylabel('loss')
    plt.bar(
        channels, loss_train, alpha=0.9, width=0.35, facecolor='lightblue', edgecolor='white', label='train'
    )
    plt.bar(
        channels + 0.35 , loss_test,  alpha=0.9, width=0.35, facecolor='yellowgreen', edgecolor='white', label='valid'
    )
    plt.legend(loc='upper right')

    # channels = np.arange(33) + 1

    # acc_test   = np.load(os.path.join(workpath, resultdir, 'exp_acc_test.npy'))
    # plt.figure(figsize=(5, 4))
    # plt.ylim(0, 100)
    # plt.title('single channel(pca&svm)'); plt.xlabel('channels'); plt.ylabel('accuracy/%')
    # plt.bar(
    #     channels, acc_test * 100,  alpha=0.9, width=0.8, facecolor='yellowgreen', edgecolor='white', label='test'
    # )

    # loss_test   = np.load(os.path.join(workpath, resultdir, 'exp_loss_test.npy'))
    # plt.figure(figsize=(5, 4))
    # plt.title('single channel(pca&svm)'); plt.xlabel('channels'); plt.ylabel('loss')
    # plt.bar(
    #     channels, loss_test,  alpha=0.9, width=0.8, facecolor='yellowgreen', edgecolor='white', label='test'
    # )







    # acc_test   = np.load(os.path.join(workpath, resultdir, 'acc_cnn.npy'))
    # plt.figure(figsize=(5, 4))
    # plt.ylim(0, 100)
    # plt.title('single channel(cnn)'); plt.xlabel('channels'); plt.ylabel('accuracy/%')
    # plt.bar(
    #     channels, acc_test * 100,  alpha=0.9, width=0.8, facecolor='yellowgreen', edgecolor='white', label='test'
    # )

    # groups = np.arange(31) + 1

    # acc_test   = np.load(os.path.join(workpath, resultdir, 'three.npy'))
    # plt.figure(figsize=(5, 4))
    # plt.ylim(0, 100)
    # plt.title('three channels(cnn)'); plt.xlabel('groups'); plt.ylabel('accuracy/%')
    # plt.bar(
    #     groups, acc_test * 100,  alpha=0.9, width=0.8, facecolor='yellowgreen', edgecolor='white', label='test'
    # )

    plt.show()

if __name__ == '__main__':
    main()