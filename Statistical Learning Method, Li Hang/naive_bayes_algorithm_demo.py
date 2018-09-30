'''
问题：计算得到的有些概率P(yc|Xi)>1
    我用朴素贝叶斯分类器写了一个能识别代码语言的小工具，但是计算联合概率的时候遇到了点问题。 
    - V2EX https://www.v2ex.com/t/190152
'''
import numpy as np
from sklearn.preprocessing import OneHotEncoder

'''
            P(yc)ΠP(Xi|yc)
P(yc|X) = ——————————————————
                ΠP(Xi)
'''

class NaiveBayes():
    def __init__(self):
        self.featureEncoder = OneHotEncoder()
        self.n_labels = None; self.n_features = None
        self.P_X = None                         # 各特征的概率              (n_features,)
        self.P_Y = None                         # 各类别的概率              (n_labels,)
        self.P_X_Y = None                       # 各类别下各特征的条件概率   (n_labels, n_features)
    def fit(self, X, y):
        X_encoded = self.featureEncoder.fit_transform(X).toarray()                  # toarray()将csr稀疏矩阵转换为稠密矩阵
        y_encoded = OneHotEncoder().fit_transform(y.reshape((-1, 1))).toarray()   # toarray()将csr稀疏矩阵转换为稠密矩阵
        self.P_X = np.mean(X_encoded, axis=0)                           # one-hot编码下，各列的均值即各特征的概率
        self.P_Y = np.mean(y_encoded, axis=0)                           # one-hot编码下，各列的均值即各了别的概率
        self.n_labels, self.n_features = y_encoded.shape[1], X_encoded.shape[1]   
        self.P_X_Y = np.zeros(shape=(self.n_labels, self.n_features))   # 各个类别下，分别统计各特征的概率
        for i in range(self.n_labels):
            X_encoded_of_yi = X_encoded[y_encoded[:, i]==1]             # 取出属于i类别的样本
            self.P_X_Y[i] = np.mean(X_encoded_of_yi, axis=0)            # one-hot编码下，各列的均值即各特征的概率
    def predict(self, X):
        X_encoded = self.featureEncoder.transform(X).toarray()
        n_samples = X_encoded.shape[0]
        y_pred_prob = np.zeros(shape=(n_samples, self.n_labels))
        for i in range(n_samples):
            for j in range(self.n_labels):
                P_Xi_encoded_Yj = X_encoded[i] * self.P_X_Y[j]          # 在Yj类别下，选出输入样本Xi对应的条件概率
                P_Xi_encoded_Yj[P_Xi_encoded_Yj==0.0] = 1.0             # 将为0值替换为1，便于求解ΠP(Xi|yc)，只要将各元素累乘即可
                P_Xi_encoded    = X_encoded[i] * self.P_X
                P_Xi_encoded[P_Xi_encoded==0.0] = 1.0
                y_pred_prob[i, j] = self.P_Y[j] * P_Xi_encoded_Yj.prod() / P_Xi_encoded.prod()
        return np.argmax(y_pred_prob, axis=1)
    def score(self, y_true, y_pred):
        ''' accuracy '''
        return np.mean(np.equal(y_true, y_pred).astype('float'))
        
if __name__ == '__main__':
    # X = [
    #     [1, 'S'],
    #     [1, 'M'],
    #     [1, 'M'],
    #     [1, 'S'],
    #     [1, 'S'],
    #     [2, 'S'],
    #     [2, 'M'],
    #     [2, 'M'],
    #     [2, 'L'],
    #     [2, 'L'],
    #     [3, 'L'],
    #     [3, 'M'],
    #     [3, 'M'],
    #     [3, 'L'],
    #     [3, 'L']
    # ]
    # y = [-1 ,-1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    
    # 注意：
    #   1. OneHotEncoder不接受字符类型的标称型数据
    #   2. OneHotEncoder不接受小于0的标称型数据
    X = [
        [1, 0],
        [1, 1],
        [1, 1],
        [1, 0],
        [1, 0],
        [2, 0],
        [2, 1],
        [2, 1],
        [2, 2],
        [2, 2],
        [3, 2],
        [3, 1],
        [3, 2],
        [3, 2],
        [3, 2]
    ]
    y = [0 ,0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    X = np.array(X); y = np.array(y)
    X_test = np.array([[2, 0], [1, 1]])

    estimator = NaiveBayes()
    estimator.fit(X, y)
    print('准确率: ', estimator.score(y, estimator.predict(X)))

    y_pred = estimator.predict(X_test)
    print(y_pred)

    '''
    准确率:  0.7333333333333333
    [0 0]
    '''