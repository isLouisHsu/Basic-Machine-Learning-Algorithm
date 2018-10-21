"""
注：单个二元分类器准确率高不代表多元分类器准确率高，
    因为二元分类器识别概率只要达到0.5以上即可，多元分类器则正确标签的概率越大越好
"""
import numpy as np
from load_data import load_X_y
from BinaryLogReg import BinaryLogisticRegression
from sklearn.cross_validation import train_test_split

class MultilabelLogisticRegression():
    """
    多元分类器
    """
    def __init__(self, n_batch=1):
        self.n_batch = n_batch
        self.n_clf = 0
        self.clfs = []   # 保存训练好的模型
    def encode(self, y):
        # 对标签进行 one-hot 编码
        n_label = len(set(list(y.reshape((-1,)))))
        ret = np.zeros(shape=(y.shape[0], n_label))
        for i in range(y.shape[0]):
            ret[i, y[i]] = 1
        return ret
    def decode(self, y):
        # 对one-hot进行解码
        return np.argmax(y, axis=0)
    def fit(self, X, y_true_label, isOneHot=False, learning_rate=0.01, max_iter=5000, min_acc=0.8):
        if isOneHot==False: y_true_label = self.encode(y_true_label)
        # 统计需要训练的二元分类器数目
        self.n_clf = y_true_label.shape[1]
        for n in range(self.n_clf):
            print("--------------------")
            print("开始训练第%d个分类器" % (n+1))
            clf = BinaryLogisticRegression(n_batch=self.n_batch)
            clf.fit(X, y_true_label[:, n], learning_rate=learning_rate, max_iter=max_iter, min_acc=min_acc)
            self.clfs.append(clf)
    def predict(self, X):
        """
        对输入的单个数据进行预测
        """
        X = np.c_[np.ones(shape=(1, 1)), X.reshape((1, -1))].reshape(-1,)
        y_pred_prob = np.zeros(shape=(self.n_clf,))
        for n in range(self.n_clf):
            y_pred_prob[n] = self.clfs[n].predict_prob(X)
        # softmax
        y_pred_prob_exp = np.exp(y_pred_prob)
        y_pred_prob_softmax = y_pred_prob_exp / np.sum(y_pred_prob_exp)
        # transfer probability to label
        y_pred_label = np.argmax(y_pred_prob_softmax)
        return y_pred_label
    def score(self, X, y_true_label, isOneHot=False):
        '''
        计算当前模型在输入的样本上的准确度
        '''
        y_pred_label = []
        for i in range(X.shape[0]):
            y_pred_label.append(self.predict(X[i]))
        y_pred_label = np.array(y_pred_label).reshape((-1, 1))   # 未编码的结果
        # 把真实标签解码
        if isOneHot==True:
            y_true_label = self.decode(y_true_label)
        # 对比是否相同
        correct = np.equal(y_pred_label, y_true_label)
        return np.mean(correct.astype('float'))

if __name__ == '__main__':
    X, y = load_X_y(one_hot=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    mdl = MultilabelLogisticRegression(n_batch=-1)
    mdl.fit(X_train, y_train, isOneHot=False, learning_rate=0.01, max_iter=5000, min_acc=0.95)

    # mdl.predict(X_train[0])
    print(mdl.score(X_train, y_train, isOneHot=False))
    print(mdl.score(X_test,  y_test,  isOneHot=False))
    pass
