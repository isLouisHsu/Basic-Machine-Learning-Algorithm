import numpy as np
import tensorflow as tf
import load_data
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

# 载入数据集
X_data, y_data_label_encode = load_data.load_X_y(one_hot=True)
X_train, X_test, y_train_label_encode, y_test_label_encode = train_test_split(X_data, y_data_label_encode, test_size=0.25, random_state=42)
# n_batch = X_train.shape[0]
n_batch = 1
batch_size = X_train.shape[0]//n_batch

# 定义一个三层的神经网络结构
# -- 神经网络的各层维度
n_input = X_data.shape[1]
n_hidden = 25
n_out = y_data_label_encode.shape[1]
# -- feedin
X = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
y = tf.placeholder(dtype=tf.float32, shape=[None, n_out])
# -- 初始化参数
W1_init, W2_init, b1_init, b2_init = load_data.load_weight()
# W1 = tf.Variable(initial_value=W1_init, dtype=tf.float32)   # n_input x n_hidden
# W2 = tf.Variable(initial_value=W2_init, dtype=tf.float32)   # n_hidden x n_out
# b1 = tf.Variable(initial_value=b1_init, dtype=tf.float32)   # n_hidden
# b2 = tf.Variable(initial_value=b2_init, dtype=tf.float32)   # n_out
W1 = tf.Variable(initial_value=tf.truncated_normal([n_input, n_hidden], stddev=0.1), dtype=tf.float32)   # n_input x n_hidden
W2 = tf.Variable(initial_value=tf.truncated_normal([n_hidden, n_out], stddev=0.1), dtype=tf.float32)   # n_hidden x n_out
b1 = tf.Variable(initial_value=tf.truncated_normal([n_hidden,], stddev=0.1), dtype=tf.float32)   # n_hidden
b2 = tf.Variable(initial_value=tf.truncated_normal([n_out,], stddev=0.1), dtype=tf.float32)   # n_out
# -- 前向传播
h1 = tf.matmul(X, W1) + b1
z1 = tf.sigmoid(h1)
h2 = tf.matmul(z1, W2) + b2
z2 = tf.sigmoid(h2)
y_pred_prob = tf.nn.softmax(z2)
# -- 损失函数
loss_cross_entropy = - tf.reduce_mean(tf.reduce_sum(y * tf.log(y_pred_prob), axis=0))
# -- 训练
train_step = tf.train.AdamOptimizer().minimize(loss_cross_entropy)
# -- 统计准确率
y_true_label_decode = tf.argmax(y, axis=1)
y_pred_label_decode = tf.argmax(y_pred_prob, axis=1)
correct = tf.equal(y_true_label_decode, y_pred_label_decode)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
# -- 准确率统计
train_acc = []
test_acc = []

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    max_iter = 5000; min_acc = 0.95
    n_iter = 0; acc_train = 0
    isDone = False
    while (n_iter < max_iter) and (not isDone):
        n_iter += 1
        # -------------- 一次迭代 --------------
        for n in range(n_batch):
            sta, end = n * batch_size, (n + 1) * batch_size
            X_train_batch = X_train[sta: end]
            y_train_label_encode_batch = y_train_label_encode[sta: end]
            sess.run(train_step, feed_dict={X: X_train_batch, y: y_train_label_encode_batch})
            acc_train = sess.run(accuracy, feed_dict={X: X_train, y: y_train_label_encode})
            acc_test = sess.run(accuracy, feed_dict={X: X_test, y: y_test_label_encode})
            train_acc.append(acc_train); test_acc.append(acc_test)
            if acc_train > min_acc:
                print('第%d次迭代, 第%d批数据, 总体训练样本准确率为:%f' % (n_iter, n, acc_train))
                isDone = True; break
        # -------------------------------------
        if n_iter%100==0:
            print('第%d次迭代, 在总体训练样本上的准确率: %f' % (n_iter, acc_train))
    if acc_train <= min_acc:
        print("超过迭代次数, 当前总体样本准确率为: ", acc_train)
    # todo
    # acc_test = sess.run(accuracy, feed_dict={X: X_test, y: y_test_label_encode})
    print("测试集上准确率为: ", acc_test)

# -- 准确度曲线
train_acc = np.array(train_acc); test_acc = np.array(test_acc)
fig = plt.figure(0)
plt.plot(np.arange(train_acc.shape[0]), train_acc, c='b')
plt.plot(np.arange(test_acc.shape[0]), test_acc, c='r')
plt.show()