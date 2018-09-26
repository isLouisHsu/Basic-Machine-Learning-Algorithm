import numpy as np
import sklearn.datasets as skdata
import sklearn.neural_network as sknn   # MLPRegressor
import sklearn.model_selection as skms
import sklearn.preprocessing as skpp

# --------------------------------------  数学函数定义  --------------------------------------------
def sigmoid(x):
    return 1/(1+np.e**(-x))
def gradSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def Linear(x):
    return x
def gradLinear(x):
    return np.ones(x.shape)
def ReLU(x):
    return np.maximum(x, 0.0)
def gradReLU(x):
    if x>0: return 1
    else: return 0
def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex)
# ------------------------------------------------------------------------------------------------
class NeuralNetwork():
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.loadData(0.2)
        self.train(0.3)
    def loadData(self, test_size):
        X, y = skdata.load_iris(True)
        X = skpp.MinMaxScaler().fit_transform(X)    # 数据缩放到(0,1)范围内
        y_label = np.zeros(shape=(y.shape[0], 3))
        for i in range(y.shape[0]):                 # 对类别进行编码
            if y[i]==0:     y_label[i] = np.array([0, 0, 1])
            elif y[i]==1:   y_label[i] = np.array([0, 1, 0])
            else:           y_label[i] = np.array([1, 0, 0])
        return skms.train_test_split(X, y_label, test_size=test_size)
    def train(self, epsilon = 0.27):
        '''
        输入层神经元个数：4， 输出层神经元个数：3
        隐含层数：1; 隐含层神经元个数：5
        激励函数：sigmoid
        '''
        # ------------------ 神经网络前向计算 -------------------
        # X为单个样本：(d0,)
        def Z1_(W1, X, b1): 
            return  X.dot(W1.T) + b1
        def H1_(f, Z1):     
            return f(Z1)
        def Z2_(W2, H1, b2):
            return H1.dot(W2.T) + b2
        def Y_(f, Z2):      
            return f(Z2)
        # ---------------------- 参数定义 ----------------------
        d0 = 4; d1 = 5; d2 = 3
        W1 = np.ones(shape=(d1, d0)); b1 = np.ones(d1)
        W2 = np.ones(shape=(d2, d1)); b2 = np.ones(d2)
        # f1 = sigmoid; gradf1 = gradSigmoid
        # f2 = sigmoid; gradf2 = gradSigmoid
        f1 = ReLU; gradf1 = gradReLU
        f2 = ReLU; gradf2 = gradReLU
        alpha = 0.01
        # -----------------------------------------------------
        while True:
            # -------------- 计算当前参数下的值 --------------
            Z1 = Z1_(W1, self.X_train, b1)
            H1 = H1_(f1, Z1)
            Z2 = Z2_(W2, H1, b2)
            Y = Y_(f2, Z2)
            e = Y - self.y_train
            # --------------- 计算损失函数的值 ---------------
            L = np.average(0.5*np.sum(e**2, axis=1))
            print(L)
            if L < epsilon: break   # 停止条件
            # ------------------- 计算梯度 ------------------
            gW1 = np.zeros(shape=(d1, d0)); gb1 = np.zeros(d1)
            gW2 = np.zeros(shape=(d2, d1)); gb2 = np.zeros(d2)
            for m in range(self.X_train.shape[0]):
                # ---------------- 输出层 ---------------- 
                for i in range(d2):
                    delta2_i = (Y[m, i]-self.y_train[m, i]) * gradf2(Z2[m, i])
                    gb2[i] += delta2_i
                    for j in range(d1):
                        gW2[i, j] += delta2_i * H1[m, j]
                # ---------------- 隐含层层 ---------------- 
                for i in range(d1):
                    delta1_i = 0
                    for d in range(d2):
                        delta2_d = (Y[m, d]-self.y_train[m, d]) * gradf2(Z2[m, d])
                        delta1_i += delta2_d * W2[d, i] * gradf1(Z1[m, i])
                    gb1[i] += delta1_i
                    for j in range(d0):
                        gW1[i, j] += delta1_i * self.X_train[m, j]
            gW1 /= m; gW2 /= m; gb1 /= m; gb2 /= m
            # ------------------- 更新参数 ------------------
            W1 -= alpha * gW1; W2 -= alpha * gW2
            b1 -= alpha * gb1; b2 -= alpha * gb2
        # -------------- 对test数据集进行验证 ---------------
        Z1 = Z1_(W1, self.X_test, b1)
        H1 = H1_(f1, Z1)
        Z2 = Z2_(W2, H1, b2)
        Y = Y_(f2, Z2)
        for i in range(Y.shape[0]):
            Y[i] = softmax(Y[i])
            maxidx = np.argmax(Y[i])
            for j in range(Y.shape[1]):
                if j==maxidx:   Y[i, j] = 1
                else:           Y[i, j] = 0
        accurate = 0
        for i in range(Y.shape[0]):
            if (Y[i]==self.y_test[i]).all():
                accurate += 1
        print("准确度：", accurate/Y.shape[0])







        
if __name__ == '__main__':
    NN = NeuralNetwork()
    NN.train()
    # x = np.array([[-1., -2., 0, 1, 2., -1], [-1, -2, 0, 1, 2., -1]])
    # print(gradReLU(x))