import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from datasets2 import workpath, load_train, load_test, get_time

from sklearn.pipeline import Pipeline 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA
from sklearn.svm import SVC
"""
class sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, 
                        coef0=0.0, shrinking=True, probability=False, 
                        tol=0.001, cache_size=200, class_weight=None, 
                        verbose=False, max_iter=-1, decision_function_shape=’ovr’, 
                        random_state=None)
"""

"""
实验三: 每次减少一个频段识别，第0通道为灰度图，该次全部通道训练
"""

dsize = (18, 22)
resultdir = os.path.join(workpath, 'results')

def main():
    # 载入数据集
    X_train, y_train = load_train(dsize)
    X_test , y_test  = load_test(dsize)
    n_samples = X_train.shape[0]; n_channels = X_train.shape[-1]

    # 定义超参数搜索
    steps = [
        ('standard', StandardScaler()),
        ('reduce_dim', PCA(n_components=int(18*22*33*0.5))), 
        ('classification', SVC(probability=True, kernel = 'poly')),
    ]
    pipe = Pipeline(steps=steps)

    hyper_params = {
        'classification__C': [0.2*i for i in range(1, 20)],
        'classification__degree': [_ for _ in range(1, 4)],
    }
    searcher = RandomizedSearchCV(pipe, hyper_params, scoring='neg_log_loss')

    # 开始训练
    acc_train_saved  = []; acc_test_saved  = []
    loss_train_saved = []; loss_test_saved = []
    logf = open(os.path.join(resultdir, 'exp3_log.txt'), 'w+')

    groupsplit = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for i_channels in range(n_channels):
        if i_channels == 0:
            X_train_i_channels = X_train.copy()
            X_test_i_channels  = X_test.copy()
        else:
            X_train_i_channels = np.delete(X_train, i_channels, axis=3)
            X_test_i_channels  = np.delete(X_test, i_channels, axis=3)
            
        X_train_i_channels = X_train_i_channels.reshape((X_train_i_channels.shape[0], -1))
        X_test_i_channels  = X_test_i_channels.reshape((X_test_i_channels.shape[0],   -1))
        
        splitRes = groupsplit.split(X_train_i_channels, y_train)
        for train_index, test_index in splitRes:
            X_train_, y_train_ = X_train_i_channels[train_index], y_train[train_index]
            X_test_,  y_test_  = X_train_i_channels[test_index],  y_train[test_index]
            
            searcher.fit(X_train_, y_train_)
            print(searcher.best_params_)

            y_train_pred_proba = searcher.predict_proba(X_train_)
            y_test_pred_proba = searcher.predict_proba(X_test_)
            y_train_pred = searcher.predict(X_train_)
            y_test_pred = searcher.predict(X_test_)

            acc_train = accuracy_score(y_train_pred, y_train_)
            acc_test  = accuracy_score(y_test_pred,  y_test_ )
            loss_train = log_loss(y_train_, y_train_pred_proba) / y_train_.shape[0]    
            loss_test  = log_loss(y_test_,  y_test_pred_proba)  / y_test_.shape[0]
            
            acc_train_saved.append(acc_train); acc_test_saved.append(acc_test)
            loss_train_saved.append(loss_train); loss_test_saved.append(loss_test)
            
            print_log = get_time() + ' channel: {:>2}/{:>2} | acc_train: {:>.2%}, loss_train:{:>.6f} | acc_test: {:>.2%}, loss_test:{:>.6f}'.\
                            format(i_channels, n_channels, acc_train, loss_train, acc_test, loss_test)
            print(print_log); logf.write(print_log + '\n')
            
            # 在测试集上预测
            y_test_i_channels_pred = searcher.best_estimator_.predict(X_test_i_channels)
            print_log = 'test: accuracy score is {:>.2%}\n'.\
                            format(accuracy_score(y_test, y_test_i_channels_pred)) + str(y_test) + '\n' + str(y_test_i_channels_pred)
            print(print_log); logf.write(print_log + '\n')
        
        print_log = '======================================================================================================='
        print(print_log); logf.write(print_log + '\n')

    logf.close()
    # 保存准确率、损失值
    np.save(os.path.join(resultdir, 'exp3_acc_train.npy'),  np.array(acc_train_saved))
    np.save(os.path.join(resultdir, 'exp3_acc_test.npy'),   np.array(acc_test_saved))
    np.save(os.path.join(resultdir, 'exp3_loss_train.npy'), np.array(loss_train_saved))
    np.save(os.path.join(resultdir, 'exp1_loss_test.npy'), np.array(loss_test_saved))
    
    


if __name__ == '__main__':
    main()
