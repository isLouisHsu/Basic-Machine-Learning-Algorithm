import numpy as np
import mathfunc as mf
from load_data import load_X_y
from sklearn.cross_validation import train_test_split

class BinaryLogisticRegression():
    """
    二元分类器
    """
    def __init__(self, n_ploy=1, n_batch=1):
        '''
        n_batch: 取值为-1时为SGD，取值为1时为批处理梯度下降
        '''
        self.n_ploy = n_ploy
        self.n_batch = n_batch
        self.theta = None
    def fit(self, X, y_true, learning_rate=0.01, max_iter = 5000, min_acc = 0.8):
        # --------------- 数据预处理部分 ---------------
        # 构造高次特征：在二维数据时适用
        if self.n_ploy > 1:
            for i in range(self.n_ploy + 1):
                n1, n2 = n_ploy - i, i
                X = np.c_[X, (X[:, 0]**n1)*(X[:, 1]**n2)]
        # 加入全1列, (n, k+1)
        X = np.c_[np.ones(shape=(X.shape[0]),), X]
        # ---------------- 参数迭代部分 ----------------
        # 初始化参数
        self.theta = np.random.uniform(-1, 1, size=(X.shape[1],))
        # 数据批次
        n_batch = X.shape[0] if self.n_batch==-1 else self.n_batch
        batch_size = X.shape[0] // n_batch
        # 停止条件
        n_iter = 0; acc = 0
        # 开始迭代
        while n_iter < max_iter:
            n_iter += 1
            for n in range(n_batch):
                n1, n2 = n*batch_size, (n+1)*batch_size
                X_batch = X[n1: n2]; y_true_batch = y_true[n1: n2]
                ####
                grad = self.lossFunctionDerivative(X_batch, y_true_batch)
                self.theta -= learning_rate * grad
                ####
                acc = self.score(X, y_true)
                if acc > min_acc:
                    print('第%d次迭代, 第%d批数据' % (n_iter, n))
                    print("当前总体样本准确率为: ", acc)
                    return self.theta
            if n_iter%100 == 0:
                print('第%d次迭代' % n_iter)
                print('准确率: ', acc)
        print("超过迭代次数")
        print("当前总体样本准确率为: ", acc)
        return self.theta
    def predict(self, X):
        """
        对输入的样本进行预测
        """
        return self.prob2label(mf.sigmoid(X.dot(self.theta)))
    def predict_prob(self, X):
        '''
        对输入的样本进行预测
        X: NumPy array, X.shape=(n, k+1)
        '''
        return mf.sigmoid(X.dot(self.theta))
    def prob2label(self, y_pred_prob):
        '''
        由predict_prob计算得概率，计算预测标签
        '''
        y_pred_label = np.sign(y_pred_prob - 0.5)
        y_pred_label[y_pred_label<0] = 0
        return y_pred_label
    def score(self, X, y_true_label):
        '''
        计算当前模型在输入的样本上的准确度
        '''
        y_pred_prob = self.predict_prob(X)
        y_pred_label = self.prob2label(y_pred_prob)
        correct = np.equal(y_pred_label, y_true_label)
        return np.mean(correct.astype('float'), axis=0)
    def lossFunction(self, y_pred_prob, y_true):
        '''
        未使用
        计算损失值: Cross Entropy
        y_pred_prob, y_true: NumPy array, shape=(n,)
        '''
        tmp = y_true*np.log(y_pred_prob) + (1 - y_true)*np.log(1 - y_pred_prob)
        return np.mean(-tmp)
    def lossFunctionDerivative(self, X, y_true):
        '''
        计算损失函数对参数theta的梯度
        对theta[j]的梯度为：(y_pred - y_true)*x[j]
        '''
        err = self.predict_prob(X) - y_true
        return X.T.dot(err)/y_true.shape[0]
        
if __name__ == '__main__':
    X, y = load_X_y(one_hot=True)
    y = y[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    mdl = BinaryLogisticRegression(n_batch=-1)
    mdl.fit(X_train, y_train, learning_rate=0.01, max_iter = 5000, min_acc = 0.95)
