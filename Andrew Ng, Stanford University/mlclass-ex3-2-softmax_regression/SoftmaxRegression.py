import numpy as np
import matplotlib.pyplot as plt
from load_data import load_X_y
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def softmax(X):
    """ softmax函数
    @param {ndarray} X: shape(batch_size, n_labels)
    """
    # 稳定数值计算
    X_max = np.max(X, axis=1).reshape((-1, 1))      # 每行的最大值
    X = X - X_max	                                # 每行减去最大值
    X = np.exp(X)
    return X / np.sum(X, axis=1).reshape((-1, 1))
    


class SoftmaxRegression():
    """
    Attributes:
        ---------
        lr: {float}
        W:  {ndarray} [w0, ..., w_9], w_i = [w_{i1}, ..., w_{i, n+1}]^T
            shape(n_features+1, n_labels)
    """
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.W = None	                                # 模型参数
        self.n_features = None
        self.n_labels = None
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder()
    def fit(self, X_train, X_valid, y_train, y_valid, min_acc=0.95, max_epoch=20, batch_size=20):
        """ 训练
        """
        # 添加首1列，输入到偏置w0
        X_train = np.c_[np.ones(shape=(X_train.shape[0],)), X_train]
        X_valid = np.c_[np.ones(shape=(X_valid.shape[0],)), X_valid]
        X_train = self.scaler.fit_transform(X_train)    # 尺度归一化
        X_valid = self.scaler.transform(X_valid)        # 尺度归一化
        self.encoder.fit(y_train.reshape(-1, 1))
        self.n_features = X_train.shape[1]
        self.n_labels = self.encoder.transform(y_train).shape[1]
        # 初始化参数
        self.W = np.random.normal(loc=0, scale=1.0, size=(self.n_features, self.n_labels))
        n_batch = X_train.shape[0] // batch_size
        # 可视化相关
        plt.ion()
        # plt.figure('loss'); plt.figure('accuracy')
        loss_train_epoch = []; loss_valid_epoch = []
        acc_train_epoch = [];  acc_valid_epoch = []
        for i_epoch in range(max_epoch):
            for i_batch in range(n_batch):              # 批处理梯度下降
                n1, n2 = i_batch * batch_size, (i_batch + 1) * batch_size
                X_train_batch, y_train_batch = X_train[n1: n2], y_train[n1: n2]
                # 预测
                y_prob_train = self.predict(X_train_batch, preprocessed=True)
                # 计算损失
                loss_train_batch = self.crossEnt(y_train_batch, y_prob_train)
                # 计算准确率
                y_label_train = np.argmax(y_prob_train, axis=1)
                a = y_train_batch.reshape((-1,))
                acc_train_batch = np.mean((y_label_train == y_train_batch.reshape((-1,))).astype('float'))
                # 计算梯度 dW
                dW = self.grad(X_train_batch, y_train_batch, y_prob_train)
                # 更新参数
                self.W -= self.lr * dW
                # 相关参数可视化
                print_log = 'epoch: {:>2}/{:>2} | batch: {:>3}/{:>3} || loss_train_batch: {:.3f}, acc_train_batch: {:.3f}'.\
                            format(i_epoch + 1, max_epoch, i_batch + 1, n_batch, loss_train_batch, acc_train_batch)
                print(print_log)
            
            # 模型验证部分
            ## 训练集
            y_prob_train = self.predict(X_train, preprocessed=True)
            loss_train = self.crossEnt(y_train, y_prob_train)
            loss_train_epoch.append(loss_train)
            y_train_pred = np.argmax(y_prob_train, 1)
            acc_train = np.mean((y_train_pred == y_train.reshape((-1,))).astype('float'))
            acc_train_epoch.append(acc_train)
            ## 验证集
            y_prob_valid = self.predict(X_valid, preprocessed=True)
            loss_valid = self.crossEnt(y_valid, y_prob_valid)
            loss_valid_epoch.append(loss_valid)
            y_valid_pred = np.argmax(y_prob_valid, 1)
            acc_valid  = np.mean((y_valid_pred == y_valid.reshape((-1,))).astype('float'))
            acc_valid_epoch.append(acc_valid)
            ## 绘图
            plt.cla()
            plt.figure('loss')
            plt.plot(np.arange(i_epoch + 1), loss_train_epoch, c='b')
            plt.plot(np.arange(i_epoch + 1), loss_valid_epoch, c='r')
            plt.figure('accuracy')
            plt.plot(np.arange(i_epoch + 1), acc_train_epoch, c='b')
            plt.plot(np.arange(i_epoch + 1), acc_valid_epoch, c='r')
            plt.pause(0.1)

            if (acc_train + acc_valid) / 2 > min_acc:
                print_log = "finish training!"
                plt.figure('loss')
                plt.savefig('loss.jpg')
                plt.figure('accuracy')
                plt.savefig('accuracy.jpg')
                return self.W

    def crossEnt(self, y_label_true, y_prob_pred):
        """ 计算交叉熵损失函数
        @param {ndarray} y_label_true: 真实标签 shape(batch_size,)
        @param {ndarray} y_prob_pred: 预测输出 shape(batch_size, n_labels)
        """
        mask = self.encoder.transform(y_label_true.reshape(-1, 1)).toarray()  # shape(batch_size, n_labels)
        y_prob_masked = np.sum(mask * y_prob_pred, axis=1)          # 每行真实标签对应的预测输出值
        y_prob_masked[y_prob_masked==0.] = 1.
        y_loss = np.log(y_prob_masked)
        loss = - np.mean(y_loss)                                    # 求各样本损失的均值
        return loss
    def grad(self, X_train, y_train, y_prob_pred):
        """ 计算梯度 \frac {∂L} {∂W_{pq}}
        @param X_train: 训练集特征
        @param y_train: 训练集标签
        @param y_prob_pred:  训练集预测概率输出
        @param y_label_pred: 训练集预测标签输出
        """
        y_train = self.encoder.transform(y_train)
        dW = X_train.T.dot(y_prob_pred - y_train)
        return dW
    def predict(self, X, preprocessed=False):
        """ 对输入的样本进行预测，输出标签
        @param {ndarray} X: shape(batch_size, n_features)
        @return {ndarray} y_prob: probability, shape(batch_size, n_labels)
                {ndarray} y_label: labels, shape(batch_size,)
        """
        if not preprocessed:
            X = np.c_[np.ones(shape=(X.shape[0],)), X]              # 添加首1项，输入到偏置w0
        X = self.scaler.transform(X)

        y_prob = softmax(X.dot(self.W))                             # 预测概率值 shape(batch_size, n_labels)
        return y_prob

if __name__ == "__main__":
    # 载入数据集
    X, y = load_X_y(one_hot=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

    estimator = SoftmaxRegression(learning_rate=0.01)
    estimator.fit(X_train, X_valid, y_train, y_valid, max_epoch=300, min_acc=0.95, batch_size=X_train.shape[0])

    
