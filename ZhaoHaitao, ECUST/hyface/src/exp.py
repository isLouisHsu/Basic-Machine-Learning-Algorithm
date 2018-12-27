import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.pipeline import Pipeline 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from datasets2 import resize, get_time, brightcontrast
from scipy.io import loadmat

(H, W, C) = (220, 180, 33)
workpath = 'F:/hyperspectral_face/'
sourcedir = 'SampleImages_F'
resultdir = 'results'

def load_train(dsize=None):
    datadir = sourcedir
    n_subjects = 25 + 1;    # 第三个人未使用
    n_sessions = 3
    n_samples = n_subjects * n_sessions	

    datapath = os.path.join(workpath, datadir)
    if not os.path.exists(datapath): print("directory doesn't exist!"); return

    if dsize is not None: (w, h) = dsize
    else: (w, h) = (W, H)

    X_train = np.zeros(shape=(n_samples, h, w, C))
    y_train = np.zeros(shape=(n_samples))
    for i_subjects in range(n_subjects):
        for i_sessions in range(n_sessions):
            i_samples = i_subjects * n_sessions + i_sessions
            matpath = os.path.join(
                    workpath, datadir, 'HyperFaceCube_F_{:d}_{:d}.mat'.\
                            format(i_subjects + 1, i_sessions + 1))
            matdata = loadmat(matpath)['SpecCube_F']
            if dsize is not None: matdata = resize(matdata, dsize)
            X_train[i_samples] = matdata
            y_train[i_samples] = i_subjects
    return X_train, y_train

def load_test(dsize=None):
    datadir = sourcedir
    n_subjects = 25 + 1	#第三个人未使用
    i_sessions = 3
    n_samples = n_subjects

    datapath = os.path.join(workpath, datadir)
    if not os.path.exists(datapath): print("directory doesn't exist!"); return

    if dsize is not None: (w, h) = dsize
    else: (w, h) = (W, H)

    X_test = np.zeros(shape=(n_samples, h, w, C))
    y_test = np.zeros(shape=(n_samples))
    for i_subjects in range(n_subjects):
        i_samples = i_subjects
        matpath = os.path.join(
                workpath, datadir, 'HyperFaceCube_F_{:d}_{:d}.mat'.\
                        format(i_subjects + 1, i_sessions + 1))
        matdata = loadmat(matpath)['SpecCube_F']
        if dsize is not None: matdata = resize(matdata, dsize)
        X_test[i_samples] = matdata
        y_test[i_samples] = i_subjects
    return X_test, y_test

def main():
    dsize = (18, 22)
    X_train, y_train = load_train(dsize)
    X_test,  y_test  = load_test(dsize)
    # 去掉第三个人
    X_train = np.delete(X_train, [6, 7, 8], axis=0)
    y_train = np.delete(y_train, [6, 7, 8], axis=0)
    X_test = np.delete(X_test, 2, axis=0)
    y_test = np.delete(y_test, 2, axis=0)

    n_samples = X_train.shape[0]; n_channels = X_train.shape[-1]

    # 定义模型
    steps = [
        ('standard', StandardScaler()),
        ('reduce_dim', PCA(n_components=int(18*22*0.5))), 
        ('classification', SVC(probability=True, kernel = 'poly')),
    ]
    pipe = Pipeline(steps=steps)

    hyper_params = {
        'reduce_dim__n_components': [350, 300, 250, 200, 150],
        'classification__C': [0.2*i for i in range(5, 20)],
        'classification__kernel': ['linear', 'poly'],
        'classification__degree': [_ for _ in range(1, 4)],
    }
    searcher = RandomizedSearchCV(pipe, hyper_params, scoring='neg_log_loss')

    # 训练、验证
    acc_train_saved  = []; acc_test_saved  = []
    loss_train_saved = []; loss_test_saved = []
    logf = open(os.path.join(workpath, resultdir, 'exp_log.txt'), 'w+')

    # for i_channels in range(n_channels):
    #     X_train_i_channels = X_train[:, :, :, i_channels]
    #     X_test_i_channels = X_test[:, :, :, i_channels]
        
    #     X_train_i_channels = X_train_i_channels.reshape((X_train_i_channels.shape[0], -1))
    #     X_test_i_channels  = X_test_i_channels.reshape ((X_test_i_channels.shape[0],  -1))

    #     searcher.fit(X_train_i_channels, y_train)
    #     print(searcher.best_params_)
            
    #     y_train_pred_proba = searcher.predict_proba(X_train_i_channels)
    #     y_test_pred_proba = searcher.predict_proba(X_test_i_channels)
    #     y_train_pred = searcher.predict(X_train_i_channels)
    #     y_test_pred = searcher.predict(X_test_i_channels)

    #     acc_train = accuracy_score(y_train_pred, y_train)
    #     acc_test  = accuracy_score(y_test_pred,  y_test)
    #     loss_train = log_loss(y_train, y_train_pred_proba) / X_train_i_channels.shape[0]    
    #     loss_test  = log_loss(y_test,  y_test_pred_proba)  / X_test_i_channels.shape[0]
            
    #     acc_train_saved.append(acc_train); acc_test_saved.append(acc_test)
    #     loss_train_saved.append(loss_train); loss_test_saved.append(loss_test)
            
    #     print_log = get_time() + ' channel: {:>2}/{:>2} | acc_train: {:>.2%}, loss_train:{:>.6f} | acc_test: {:>.2%}, loss_test:{:>.6f}'.\
    #                     format(i_channels, n_channels, acc_train, loss_train, acc_test, loss_test)
    #     print(print_log); logf.write(print_log + '\n')
        
    #     print_log = '======================================================================================================='
    #     print(print_log); logf.write(print_log + '\n')
    # logf.close()
    
    # # 保存准确率、损失值
    # np.save(os.path.join(workpath, resultdir, 'exp_acc_train.npy'),  np.array(acc_train_saved))
    # np.save(os.path.join(workpath, resultdir, 'exp_acc_test.npy'),   np.array(acc_test_saved))
    # np.save(os.path.join(workpath, resultdir, 'exp_loss_train.npy'), np.array(loss_train_saved))
    # np.save(os.path.join(workpath, resultdir, 'exp_loss_test.npy'), np.array(loss_test_saved))

    contrastAdjust = [_*0.1+0.5 for _ in range(10)]
    for adjval in contrastAdjust:
        X_train_adj = brightcontrast(X_train, 0, adjval)
        # X_test_adj  = brightcontrast(X_test, 0, adjval) 

        X_train_adj = X_train_adj.reshape((X_train_adj.shape[0], -1))
        X_test_adj  = X_test_adj.reshape((X_test_adj.shape[0],   -1))
        
        searcher.fit(X_train_adj, y_train)
        print(searcher.best_params_)
            
        y_train_pred_proba = searcher.predict_proba(X_train_adj)
        y_test_pred_proba = searcher.predict_proba(X_test_adj)
        y_train_pred = searcher.predict(X_train_adj)
        y_test_pred = searcher.predict(X_test_adj)

        acc_train = accuracy_score(y_train_pred, y_train)
        acc_test  = accuracy_score(y_test_pred,  y_test )
        loss_train = log_loss(y_train, y_train_pred_proba) / y_train.shape[0]    
        loss_test  = log_loss(y_test,  y_test_pred_proba)  / y_test.shape[0]
            
        acc_train_saved.append(acc_train); acc_test_saved.append(acc_test)
        loss_train_saved.append(loss_train); loss_test_saved.append(loss_test)
            
        print_log = get_time() + ' adjust: {:>.2f} | acc_train: {:>.2%}, loss_train:{:>.6f} | acc_test: {:>.2%}, loss_test:{:>.6f}'.\
                            format(adjval, acc_train, loss_train, acc_test, loss_test)
        print(print_log); logf.write(print_log + '\n')

    logf.close()
    # 保存准确率、损失值
    np.save(os.path.join(resultdir, 'exp_adj_acc_train.npy'),  np.array(acc_train_saved))
    np.save(os.path.join(resultdir, 'exp_adj_acc_test.npy'),   np.array(acc_test_saved))
    np.save(os.path.join(resultdir, 'exp_adj_loss_train.npy'), np.array(loss_train_saved))
    np.save(os.path.join(resultdir, 'exp_adj_loss_test.npy'), np.array(loss_test_saved))

if __name__ == '__main__':
    main()