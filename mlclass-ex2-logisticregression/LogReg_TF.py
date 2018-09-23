"""
问题：迭代过程中，参数变化非常小
"""
import numpy as np
import tensorflow as tf

# Training set for the first half of the exercise
# sigmoid(θ0*1 + θ1*x1 + θ2*x2)
# θ2*x2 = -θ0 - θ1*x1
file1 = './ex2data1.txt'
# Training set for the second half of the exercise
file2 = './ex2data2.txt'

def loadData(file, n_ploy=1, display=False):
    '''
    读取数据，添加全1列，返回X, y
    注：读取的数据都只有两个特征，故构造高次时只需考虑两列向量即可
    '''
    datalist = []
    with open(file) as f:
        dataline = f.readline()
        while(dataline):
            datalist.append(list(eval(dataline)))
            dataline = f.readline()
    dataarray = np.array(datalist)
    X = dataarray[:, :-1]; y = dataarray[:, -1].reshape((-1, 1))
    # 构造高次特征
    if n_ploy > 1:
        for i in range(n_ploy + 1):
            n1, n2 = n_ploy - i, i
            X = np.c_[X, (X[:, 0]**n1)*(X[:, 1]**n2)]
    # 加入全1列，便于计算参数
    X = np.c_[np.ones(shape=(X.shape[0]),), X]    
    if display==True:
        plt.figure('origin_data')
        plt.scatter(X[:, 1], X[:, 2], c=y)
        plt.show()
    return X, y
##############################################################
# 载入数据集，制定相关参数
X, y_true = loadData(file1, n_ploy=1, display=False)
n_dim = X.shape[1]
batch_size = X.shape[0]
n_batch = X.shape[0]//batch_size

# 定义两个feedin, 不限长度
x = tf.placeholder(tf.float32, shape=[None, n_dim])
y_true_label = tf.placeholder(tf.float32, shape=[None, 1])

# 定义Logistics回归模型
# 注: tensorflow内置激活函数tf.sigmoid()
theta = tf.Variable(tf.truncated_normal(shape=[n_dim, 1], stddev=0.1))
y_pred_prob = tf.sigmoid(tf.matmul(x, theta))

# 将计算得概率(0.0, 1.0)转换为预测标签[0, 1], 以0.5为阈值
# 注: tensorflow内置激活函数tf.nn.relu(), 即max(0, x)
y_pred_label = tf.nn.relu(tf.sign(y_pred_prob - 0.5))

# 计算准确度
# 注: 若采用ont-hot编码，则需转换为标签值: y_pred_label = tf.argmax(y_pred_label_onehot, axis=1)
correct_prediction = tf.equal(y_pred_label, y_true_label)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 定义损失函数
# 注: 若采用one-hot编码，crossEnt = - tf.reduce_mean(y_true_label_onehot * y_pred_prob_softmax)
crossEnt = - tf.reduce_mean(y_true_label * y_pred_prob + (1 - y_true_label) * (1 - y_pred_prob))
# 以下为使用tensorflow内置函数tf.nn.sigmoid_cross_entropy_with_logits()计算交叉熵
# crossEnt = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred_prob, labels=y_true_label))

# 使用梯度下降法优化
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(crossEnt)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    max_iter = 10000; min_acc = 0.9
    accuracy = 0.; n_iter = 0; isDone = False
    while (n_iter < max_iter) and (not isDone):
        n_iter += 1
        for n in range(n_batch):
            x_batch = X[n*batch_size: (n+1)*batch_size]
            y_true_batch = y_true[n*batch_size: (n+1)*batch_size]
            sess.run(train_step, feed_dict={x: x_batch, y_true_label: y_true_batch})
            accuracy = sess.run(acc, feed_dict={x: X, y_true_label: y_true})
            if accuracy > min_acc: 
                isDone = True
                print('第%d次迭代, 第%d批数据' % (n_iter, n))
                print("当前总体样本准确率为: ", accuracy)
                print("当前参数值为: ", sess.run(theta))
                break
            if n_iter%100 == 0:
                print('第%d次迭代' % n_iter)
                print('准确率： ', accuracy)
                print("当前参数值为: ", sess.run(theta))
        if n_iter >= max_iter:
            print("超过迭代次数")
            print("当前总体样本准确率为: ", accuracy)
            print("当前参数值为: ", sess.run(theta))
