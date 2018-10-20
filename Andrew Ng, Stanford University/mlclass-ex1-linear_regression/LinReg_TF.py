import numpy as np
import tensorflow as tf

# Linear regression with one variable
# y = θ0 + θ1*x
file1 = './ex1data1.txt'
# Linear regression with multiple variables
# y = θ0 + θ1*x1 + θ2*x2
file2 = './ex1data2.txt'

def loadData(file):
    datalist = []
    with open(file) as f:
        dataline = f.readline()                     # 读取一行数据(字符串)
        while dataline:
            datalist.append(list(eval(dataline)))   # 保存到2维list
            dataline = f.readline()
    dataarray = np.array(datalist)                  # list转换为array
    X_y = np.c_[np.ones(shape=(dataarray.shape[0]),), dataarray]    # 加入全1列，便于计算参数
    X = X_y[:, :-1]; y = X_y[:, -1].reshape(-1, 1)
    # 进行尺度缩放, 注意, 若输入测试集数据，需保存max_和min_进行相同的尺度缩放
    # for i in range(1, X.shape[1]):
    #     max_ = np.max(X[:, i]); min_ = np.min(X[:, i])
    #     X[:, i] = (X[:, i] - min_)/(max_-min_)
    return X, y

# 载入数据, 指定相关参数
X, y = loadData(file1)
n_dim = X.shape[1]
n_batch = 1; batch_size = X.shape[0] // n_batch

# 定义两个feedin, 不限长度
x = tf.placeholder(tf.float32, [None, n_dim])
y_true = tf.placeholder(tf.float32, [None, 1])

# 定义线性回归模型
theta = tf.Variable(tf.zeros(shape=[n_dim, 1]))
y_pred = tf.matmul(x, theta)

# 损失函数: MSE
loss = tf.reduce_mean(tf.square(y_pred - y_true))

# 梯度下降法求解，最小化目标函数
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    max_iter = 10000; min_loss = 9.0                                    # 初始化变量
    n_iter = 0; theEnd = False
    while((n_iter<max_iter)&(theEnd==False)):
        n_iter += 1
        for n in range(n_batch):
            x_batch = X[n*batch_size: (n+1)*batch_size]                             # 获取训练的一批数据
            y_true_batch = y[n*batch_size: (n+1)*batch_size]                        # 获取训练的一批数据
            sess.run(train_step, feed_dict={x:x_batch, y_true:y_true_batch})        # 进行优化
            lossValue = sess.run(loss, feed_dict={x:X, y_true:y})  # 计算损失函数的值
            if lossValue < min_loss:
                print('第%d次迭代, 第%d批数据' % (n_iter, n))
                print("当前总体样本损失函数值为: ", lossValue)
                print("当前参数值为: ", sess.run(theta))
                theEnd = True
    if not n_iter<max_iter:
        print("超过迭代次数，当前总体样本损失函数值为: ", lossValue)
        print("当前参数值为: ", sess.run(theta))

"""
理论最优:  [-3.89578088  1.19303364]
理论最优解下的损失值为:  4.476971375975179
file1: 
    n_batch = 1
        第573次迭代, 第0批数据
        当前总体样本损失函数值为:  8.999808
        当前参数值为:  [[-3.393881 ] [ 1.1426123]]
file2:
    n_batch = 1
    RuntimeWarning: invalid value encountered in less
    if lossValue < min_loss:
"""