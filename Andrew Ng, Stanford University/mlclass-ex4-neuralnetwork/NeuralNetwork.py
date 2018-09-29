'''
梯度计算计算量太太太太太大了
'''
import numpy as np
import load_data
from sklearn.cross_validation import train_test_split

def sigmoid(x):
    return 1 / (1 - np.exp(x))
def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork():
    """
    三层的神经网络结构，激活函数采用sigmoid
    input layer     -----------> hidden layer      ------------> output layer
                    W1<k+1, k'>                    W1<k'+1, n_out>
    (I, X)<n, k+1>  -----------> (I, Z1)<n, k'+1>  ------------> Z2<n, n_out> -> prob<n, n_out>
    输入的数据x范围为(0, 1)
    对于输入的y_true_label，需进行one-hot编码
    """
    def __init__(self, input_layer_size=400, hidden_layer_size=25, output_layer_labels=10):
        self.input_layer_size  = input_layer_size + 1;      # 20x20 Input Images of Digits, another one for bias
        self.hidden_layer_size = hidden_layer_size + 1;     # 25 hidden units, another one for bias
        self.output_layer_labels = output_layer_labels;     # 10 labels, from 1 to 10
        self.W1, self.W2, self.b1, self.b2 = load_data.load_weight()    # initialize parameters
    # ----------- one-hot -----------
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
    # --------- 网络计算基础 ---------
    def h(self, x, W, b): return W.T.dot(x) + b                         # forward propagation
    def z(self, h): return sigmoid(h)                                   # apply sigmoid as activation function
    def softmax(self, x): return np.exp(x)/np.sum(np.exp(x))            # softmax
    def prob2label(self, prob): return self.decode(prob)                # return index as label
    # --------- 神经网络算法 ---------
    def fit(self, X, y_true_label, isOneHot=False, learning_rate=0.01, n_batch=1, max_iter=5000, min_acc=0.9):
        '''
        训练神经网络的参数
        '''
        if not isOneHot: y_true_label = self.encode(y_true_label)
        n_batch = X.shape[0] if n_batch==-1 else n_batch; batch_size = X.shape[0]//n_batch
        # 停止条件
        n_iter = 0; acc = 0
        # 开始迭代
        while n_iter < max_iter:
            n_iter += 1
            for n in range(n_batch):
                n1, n2 = n*batch_size, (n+1)*batch_size
                X_batch = X[n1: n2]; y_true_batch = y_true_label[n1: n2]
                ####
                dW1, dW2 = self.dLoss(X_batch)
                W1 -= learning_rate * dW1; W2 -= learning_rate * dW2
                ####
                acc = self.score(y_true_label=y_true_label, y_pred_label=self.predict(X))  # 在总体样本上的准确度
                if acc > min_acc:
                    print('第%d次迭代, 第%d批数据' % (n_iter, n))
                    print("当前总体样本准确率为: ", acc)
                    return self.W1, self.W2
            if n_iter%100 == 0:
                print('第%d次迭代' % n_iter)
                print('准确率: ', acc)
        print("超过迭代次数")
        print("当前总体样本准确率为: ", acc)
        return self.W1, self.W2
    def predict_prob(self, X, return_all_parameters=False):
        '''
        对输入的数据进行预测，返回(n, n_output_layer)的数组
        '''
        h1 = self.h(X, self.W1, self.b1)
        z1 = self.z(h1)
        h2 = self.h(z1, self.W2, self.b2)
        z2 = self.z(h2)
        p  = self.softmax(z2)
        if return_all_parameters: return h1, z1, h2, z2, p
        return p
    def predict(self, X):
        '''
        对输入的数据进行预测
        '''
        output_label   = self.prob2label(self.predict_prob(X))
        return output_label
    def score(self, y_true_label, y_pred_label):
        '''
        计算预测的准确度
        '''
        n_samples = y_true_label.shape[0]
        isCorrect = np.equal(y_true_label.reshape(shape=(n_samples,)), y_pred_label.reshape(shape=(n_samples,)))
        return np.mean(isCorrect.astype('float'))
    def loss(self, y_true_label_onehot, y_pred_prob):
        '''
        程序中未使用
        损失函数：交叉熵
        y_pred_prob: 神经网络经softmax求出的概率值
        '''
        maxprob_samples = np.sum(y_true_label_onehot * y_pred_prob, axis=0)
        return -np.mean(np.log(maxprob_samples))
    def dLoss(self, X):
        '''
        损失函数对于参数的梯度，反向传播(BP)算法
        简单易懂的softmax交叉熵损失函数求导 - CSDN博客 
        https://blog.csdn.net/allenlzcoder/article/details/78591535
        softmax函数与交叉熵的反向梯度传导 - CSDN博客 
        https://blog.csdn.net/fireflychh/article/details/73794270
        计算较复杂，没有采用矩阵运算的方式
        '''
        dW1 = np.zeros(shape=self.W1.shape)
        dW2 = np.zeros(shape=self.W2.shape)
        db1 = np.zeros(shape=self.b1.shape)
        db2 = np.zeros(shape=self.b2.shape)
        for i in range(X.shape[0]):
            h1, z1, h2, z2, prob = self.predict_prob(X[i], return_all_parameters=True)
            # 对参数W1的梯度
            for p in range(dW1.shape[0]):
                for q in range(dW1.shape[1]):
                    for j in range(self.output_layer_labels):
                        dW1[p, q] += (prob[j] - 1) * sigmoid_d(h2[j]) * self.W2[q, j] * sigmoid_d(h1[q]) * X[i, p]
            # 对参数W2的梯度
            for p in range(dW2.shape[0]):
                for q in range(dW2.shape[1]):
                    dW1[p, q] += (prob[q] - 1) * sigmoid_d(h2[q]) * z1[p]
        dW1 /= X.shape[0]; dW2 /= X.shape[0]
        return dW1, dW2

if __name__ == '__main__':
    X, y = load_data.load_X_y(one_hot=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    mdl = NeuralNetwork(input_layer_size=400, hidden_layer_size=25, output_layer_labels=10)
    mdl.fit(X_train, y_train, isOneHot=False, learning_rate=0.01, n_batch=1, max_iter=5000, min_acc=0.9)