import numpy as np
from sklearn.cross_validation import train_test_split
from dataset import load_X_y
import matplotlib.pyplot as plt
import sklearn.datasets as sd

# ex6data1.mat - Example Dataset 1
file1 = './ex6data1.mat'
# ex6data2.mat - Example Dataset 2
file2 = './ex6data2.mat'
# ex6data3.mat - Example Dataset 3
file3 = './ex6data3.mat'

class SVC():
    '''
    实现线性SVM分类器
    由于引入核函数后进行的投射不确定，特别是引入高斯核函数后可投射到无穷维，所以未实现非线性模型的预测
    '''
    def __init__(self):
        self.weight = None
        self.bias = None
    def fit(self, X, y_true, C=1.0, delta=0.001, max_iter=20, showResult=True):
        '''
        用SMO求解
        '''
        # ---------------------------------
        alpha = np.random.uniform(size=(X.shape[0])) # 拉格朗日乘子
        bias = 0                            # 分界超平面的偏置值
        # ------------ 基础运算 ------------
        def chooseJ(i): # 返回与i不同的整数
            j = i
            while j==i: j = np.random.randint(0, X.shape[0])
            return j
        def K(i, j): 
            x1 = X[i]
            x2 = X[j]
            a = np.dot(x1, x2)
            return np.dot(X[i], X[j])
        def f(i):       # Σα[m]*y_true[m]*K(i,m) + b
            tmp = bias
            for m in range(X.shape[0]):
                tmp += alpha[m] * y_true[m] * K(i, m)
            return tmp
        def E(i):
            return f(i) - y_true[i]
        def clip(x, H, L):
            if x > H:   x = H
            elif x < L: x = L
            return x
        def weight(): 
            '''
            Σα[i]*y[i]*X[i,:]
            '''
            # alpha_y_doubled = np.tile(alpha * y_true, X.shape[1]).reshape((-1, X.shape[1]))
            # return np.sum(alpha_y_doubled * X, axis=0)ret = 0
            ret = 0
            for i in range(X.shape[0]):
                ret += alpha[i]*y_true[i]*X[i].reshape((X.shape[1], 1))
            return ret
        # -------------- 迭代 --------------
        n_iter = 0                          # 迭代次数，当乘子发生更新时，重新计数迭代
        while n_iter < max_iter:
            n_update = 0                    # 乘子更新次数计数
            for i in range(X.shape[0]):     # 选择第i个参数
                if ((E(i) * y_true[i] < -delta) and (alpha[i] < C)) or\
                    ((E(i) * y_true[i] > delta)  and (alpha[i] > 0)):   # 遍历一遍整个数据集，对每个不满足KKT条件的参数，选作第一个待修改参数
                    j = -1
                    # 第二个参数的选择：启发式规则2<1>, 寻找abs(Ei-Ej)最大的
                    # abs_Ei_Ej = -float('inf')
                    # for j_ in range(X.shape[0]):
                    #     if j_ != i:
                    #         abs_Ei_Ej_tmp = abs(E(i) - E(j_))
                    #         if abs_Ei_Ej_tmp > abs_Ei_Ej:
                    #             j = j_; abs_Ei_Ej = abs_Ei_Ej_tmp
                    # 第二个参数的选择：启发式规则2<2>, 随机选择
                    if j == -1: j = chooseJ(i)
                    # 保存旧参数
                    alpha_i_old = alpha[i]; alpha_j_old = alpha[j]
                    # 计算新参数的上下界
                    if y_true[i] != y_true[j]:
                        tmp = alpha_j_old - alpha_i_old
                        L, H = max(0, tmp), min(C, tmp + C)
                    else:
                        tmp = alpha_i_old + alpha_j_old
                        L, H = max(0, tmp - C), min(C, tmp)
                    if L == H: continue # todo
                    # 计算eta
                    eta = K(i, i) + K(j, j) - 2 * K(i, j)
                    if eta < 0: continue # todo: 讨论二次项系数的符号
                    # 更新alpha[j]
                    alpha[j] += y_true[j] * (E(i) - E(j)) / eta
                    alpha[j] = clip(alpha[j], H, L)
                    if abs(alpha_j_old - alpha[j])<0.001: 
                        # print('lda[j]变化太小')
                        continue
                    # 更新alpha[i]
                    alpha[i] += y_true[i] * y_true[j] * (alpha_j_old - alpha[j])
                    # 更新bias
                    bias_i = bias - E(i) + y_true[i]*K(i, i)*(alpha_i_old - alpha[i]) + y_true[j]*K(i, j)*(alpha_j_old - alpha[j])
                    bias_j = bias - E(j) + y_true[i]*K(j, i)*(alpha_i_old - alpha[i]) + y_true[j]*K(j, j)*(alpha_j_old - alpha[j])
                    if 0 < alpha[i] and alpha[i] < C: bias = bias_i
                    elif 0 < alpha[j] and alpha[j] < C: bias = bias_j
                    else: bias = (bias_i + bias_j) / 2.0
                    # 更新次数加1
                    n_update += 1
                    print("第%d次迭代时样本%d,%d进行了%d次优化" % (n_iter, i, j, n_update))
            if n_update == 0:   # alpha参数未更新
                n_iter += 1    
                print('第%d次迭代，无参数更新' % (n_iter))
            else: 
                # print('迭代次数重新计数')
                n_iter = 0      # alpha参数更新, 重新迭代
        self.weight, self.bias = weight(), bias
        if X.shape[1]==2 and showResult:
            plt.figure('result')
            x1 = X[:, 0]; x2 = X[:, 1]
            plt.scatter(x1, x2, c=y_true)
            X1 = np.linspace(min(x1), max(x1))
            X2 = -(self.weight[0]*X1 + self.bias)/self.weight[1]   # self.W[0]*x1 + self.W[1]*x2 + b = 0
            plt.plot(X1, X2)
            plt.show()
    
    def predict(self, X):
        a = np.sign(X.dot(self.weight) + self.bias)
        return np.sign(X.dot(self.weight) + self.bias)
    def score(self, X, y_true):
        '''
        准确度
        '''
        y_pred = self.predict(X)
        return np.mean(np.equal(y_true, y_pred).astype('float'))

        
if __name__ == '__main__':
    X, y = load_X_y(file1, display=True)
    # X, y = sd.make_classification(n_samples=50, n_features=2, n_redundant=0, n_classes=2, n_clusters_per_class=1); y[y==0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    clf = SVC()
    clf.fit(X_train, y_train, C=0.6, delta=0.001, max_iter=50, showResult=True)
    acc = clf.score(X_train, y_train)
    pass

     
