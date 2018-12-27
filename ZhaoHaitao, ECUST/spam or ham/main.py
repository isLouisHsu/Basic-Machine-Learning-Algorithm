import os
import numpy as np
import pandas as pd
from tkinter import _flatten
import re
import string
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder
from PCA import PCA

# --------------------- class ---------------------
class Words2Vector():
    '''
    建立字典，将输入的词列表转换为向量，表示各词出现的次数
    '''
    def __init__(self):
        self.dict = None
        self.n_word = None
    def fit_transform(self, words):
        self.fit(words)
        return self.transform(words)
    def fit(self, words):
        """
        @param {list[list[str]]} words
        """
        words = _flatten(words)
        words = self.filt(words)

        self.word = list(set(words))    # 去重
        self.n_word = len(set(words))   # 统计词的个数
        self.dict = dict(zip(self.word, [_ for _ in range(self.n_word)]))       # 各词在字典中的位置
    def transform(self, words):
        """
        @param {list[list[str]]} words
        @return {ndarray} retarray: normalized vector
        """
        retarray = np.zeros(shape=(len(words), self.n_word))                    # 返回的向量
        for i in range(len(words)):
            words[i] = self.filt(words[i])
        for i in range(len(words)):
            for w in words[i]:
                if w in self.word:                                              # 是否在训练集生成的字典中
                    retarray[i, self.dict[w]] += 1
        return retarray
    def filt(self, flattenWords):
        retWords = []
        en_stops = set(stopwords.words('english'))                              # 停用词列表
        for word in flattenWords:
            word = word.translate(str.maketrans('', '', string.whitespace))     # 去除空白
            word = word.translate(str.maketrans('', '', string.punctuation))    # 去除标点
            word = word.translate(str.maketrans('', '', string.digits))         # 去除数字
            if word not in en_stops and (len(word) > 1):                        # 删除停用词，并除去长度小于等于2的词
                retWords.append(word)
        return retWords

class TfidfVectorizer():
    def __init__(self):
        self.idf = None
    def fit_transform(self, num_vec):
        self.fit(num_vec)
        return self.transform(num_vec)
    def fit(self, num_vec):
        """
        @param {ndarray}: num_vec, shape(N_sample, N_feature)
        """
        num_vec[num_vec>0] = 1
        n_doc = num_vec.shape[0]
        n_term = np.sum(num_vec, axis=0)    # 各词出现过的文档次数
        self.idf = np.log((n_doc + 1) / (n_term + 1)) + 1
        a = np.isnan(self.idf).any()
        return self.idf
    def transform(self, num_vec):
        """
        @param {ndarray}: num_vec, shape(N_sample, N_feature)
        """
        # 求解词频向量，由于部分向量为空，故下句会出现问题
        # tf = num_vec / np.sum(num_vec, axis=1).reshape(-1, 1) => nan
        # 解决方法：只对非空向量进行词频计算
        tf = np.zeros(shape=num_vec.shape)
        n_terms = np.sum(num_vec, axis=1); idx = (n_terms!=0)

        tf[idx] = num_vec[idx] / n_terms[idx].reshape(-1, 1)            # 计算词频，只对非空向量进行
        
        tfidf = tf * self.idf
        tfidf[idx] /= np.linalg.norm(tfidf, axis=1)[idx].reshape(-1, 1) # 单位化，只对非空向量进行
        
        return tfidf

class SpamOrHam():
    """ based on Naive Bayes
    """
    def __init__(self):
        self.numvectorizer = Words2Vector()
        self.tfidfvectorizer = TfidfVectorizer()
        self.n_features = None
        self.priori = None
        self.likelihood_mu = None	                                 # 设似然函数p(x|c)为高斯分布
        # self.likelihood_sigma = None	                             # 设似然函数p(x|c)为高斯分布
    # def gaussian(self, x, mu, sigma):
    #     y = np.exp(- 0.5 * ((x - mu) / sigma)**2) / (np.sqrt(2*np.pi) * sigma)
    #     return y 	                                                # 容易下溢
    def multigaussian(self, x, mu):
        x = x - mu
        a = np.exp(-0.5 * x.T.dot(x))
        # b = np.sqrt(2*np.pi)**self.n_features     # => inf
        # return a / b
        return a
    def fit(self, labels, text):
        """
        @param {ndarray} labels: shape(N_samples, ), labels[i] \in {0, 1}
        @param {list[list[str]]} words
        """
        labels = self.encodeLabel(labels); words = self.text2words(text)

        vecwords = self.numvectorizer.fit_transform(words)              # 向量化
        vecwords = self.tfidfvectorizer.fit_transform(vecwords)         # tfidf, shape(N_samples, N_features)

        isnotEmpty = (np.sum(vecwords, axis=1)!=0)                      # 去掉空的样本
        vecwords = vecwords[isnotEmpty]; labels = labels[isnotEmpty]

        self.n_features = vecwords.shape[1]

        labels = OneHotEncoder().fit_transform(labels.reshape((-1, 1))).toarray()
        self.priori = np.mean(labels, axis=0)                           # 先验概率

        self.likelihood_mu = np.zeros(shape=(2, vecwords.shape[1]))	    # 设似然函数p(x|c)为高斯分布
        # self.likelihood_sigma = np.zeros(shape=(2, vecwords.shape[1]))  # 设似然函数p(x|c)为高斯分布
        for i in range(2):
            vec = vecwords[labels[:, i]==1]
            self.likelihood_mu[i] = np.mean(vec, axis=0)
        #     self.likelihood_sigma[i] = np.std(vec, axis=0)
        # self.likelihood_sigma = np.clip(self.likelihood_sigma, 1e-4, 10)
    def predict(self, text):
        """
        @param {list[list[str]]} words
        @note:
                      p(x|c)P(c)
            P(c|x) = ------------
                         p(x)
        """
        pred_porba = np.ones(shape=(len(text), 2))      
        
        words = self.text2words(text)
        vecwords = self.tfidfvectorizer.transform(
                                self.numvectorizer.transform(words))    # 向量化

        for i in range(vecwords.shape[0]):
            for c in range(2):
                pred_porba[i, c] *= self.priori[c]
                # for j in range(vecwords.shape[1]):
                #     w = vecwords[i, j]
                #     mu = self.likelihood_mu[c, j]
                #     sigma = self.likelihood_sigma[c, j]
                #     pred_porba[i ,c] *= self.gaussian(w, mu, sigma) # 这样会下溢

                pred_porba[i, c] = self.multigaussian(vecwords[i], self.likelihood_mu[c])
        pred = np.argmax(pred_porba, axis=1)
        return self.decodeLabel(pred)
    def encodeLabel(self, strLabel):
        """ encode labels
        @param {Pandas series} strLabel
        @return {Numpy ndarray} label
        """
        label = np.array(strLabel.replace('ham', 0).replace('spam', 1))
        return label

    def decodeLabel(self, numLabel):
        """ decode labels
        @param {Numpy ndarray} numLabel
        @return {Pandas series} label
        """
        label = pd.Series(numLabel).replace(0, 'ham').replace(1, 'spam')
        return label

    def text2words(self, text):
        """
        @param {Pandas series}: sentence
        @return {list[str]}: words
        """
        words = list(text)
        count = len(words)
        for i in range(count):
            words[i] = words[i].split()
        return words
    def score(self, labels_true, labels_pred):
        num_true = self.encodeLabel(labels_true)
        num_pred = self.encodeLabel(labels_pred)
        acc = np.mean((num_true==num_pred).astype('float'))
        return acc

# --------------------- data ---------------------
def load_train():
    """
    @return {Pandas series} labels, text 
    """
    filePath = "./data/train.csv"
    fileData = pd.read_csv(filePath, names=['Label', 'Text'])
    labels = fileData['Label'][1: ]; text = fileData['Text'][1: ]
    return labels, text

def load_test():
    filePath = "./data/test.csv"
    text = pd.read_csv(filePath, names=['Text'])['Text'][1: ]
    return text

def save_test_result(pred_test, filePath):
    pred_test = list(pred_test)
    with open(filePath) as f:
        for t in pred_test:
            f.writelines(t)
    print("saved all!")
# --------------------- main ---------------------------
def main():
    labels, text = load_train()                 # 载入数据，文本已切分成单词并过滤，标签为Numerical:{'ham':0, 'spam':1}
    
    estimator = SpamOrHam()
    estimator.fit(labels, text)
    pred_train = estimator.predict(text)
    
    acc = estimator.score(labels, pred_train)
    print('accuracy score is: {:2.4}%'.format(acc*100))

    text = load_test()
    pred_test = estimator.predict(text)

    savePath = "./data/sampleSubmission"
    save_test_result(pred_test, "")

if __name__ == '__main__':
    main()
