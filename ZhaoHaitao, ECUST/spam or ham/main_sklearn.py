import numpy as np
import pandas as pd
import string

import sklearn
import sklearn.preprocessing as skpre
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import PCA

dict_label = {0: 'ham', 1: 'spam'}

def drop(list_words):
    for i in range(len(list_words)):
        list_words[i] = list_words[i].translate(str.maketrans('', '', string.punctuation))    # 去除标点
        list_words[i] = list_words[i].translate(str.maketrans('', '', string.digits))         # 去除数字
    return list_words

def main():
    trainfile = "./data/train.csv"
    testfile = "./data/test.csv"
    
    # 读取原始数据
    data_train = pd.read_csv(trainfile, names=['Label', 'Text'])
    txt_train  = list(data_train['Text'])[1: ]; label_train = list(data_train['Label'])[1: ]
    drop(txt_train)                                             # 删除数字和标点
    txt_test   = list(pd.read_csv(testfile, names=['Text'])['Text'])[1: ]
    drop(txt_test)                                              # 删除数字和标点

    # 训练
    vectorizer = TfidfVectorizer(stop_words='english')          # 删除英文停用词
    vec_train = vectorizer.fit_transform(txt_train).toarray()   # 提取文本特征向量
    # reduce_dim = PCA(n_components = 4096)
    # vec_train = reduce_dim.fit_transform(vec_train)
    estimator = BernoulliNB()
    estimator.fit(vec_train, label_train)                       # 训练朴素贝叶斯模型

    # 测试
    label_train_pred = estimator.predict(vec_train)
    acc = np.mean((label_train_pred==label_train).astype('float'))
    
    # 预测
    vec_test = vectorizer.transform(txt_test).toarray()
    # vec_test = reduce_dim.transform(vec_test)
    label_test_pred = estimator.predict(vec_test)
    with open('./data/sampleSubmission.txt', 'w') as f:
        for i in range(label_test_pred.shape[0]):
            f.write(label_test_pred[i] + '\n')

if __name__ == '__main__':
    main()