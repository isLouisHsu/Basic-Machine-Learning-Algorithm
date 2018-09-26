import numpy as np
import load_data
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler

class AnomalyDetection():
    def __init__(self):
        self.mu = None      # (n_feature,)
        self.sigma = None   # (n_feature, n_feature)
        self.epsilon = None # (1,)
    def predict_prob(self, x, mu, sigma):
        '''
        @param x: 单个样本
        @note: multi-dimension Gaussian distribution
        '''
        n_dim = mu.shape[0]
        x_rm_mean = x - mu
        sigma_det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        param1 = (2 * np.pi)**(- n_dim/2)
        param2 = sigma_det**(- 0.5)
        param3 = - 0.5 * x_rm_mean.T.dot(sigma_inv).dot(x_rm_mean)
        return param1 * param2 * np.exp(param3)
    def predict(self, X):
        '''
        @param X: n个样本
        '''
        y_pred_prob  = np.zeros(shape=(X.shape[0],))
        y_pred_label = np.zeros(shape=(X.shape[0],), dtype='uint8')
        for i in range(X.shape[0]):
            y_pred_prob[i] = self.predict_prob(X[i], self.mu, self.sigma)
        y_pred_label[y_pred_prob >= self.epsilon] = 0
        y_pred_label[y_pred_prob <  self.epsilon] = 1
        return y_pred_label
    def fit(self, X_train, X_valid, y_valid):
        '''
        @param X_train: 估计高斯分布的参数
        @param X_valid, y_valid: 估计阈值
        '''
        # 估计高斯分布的参数
        self.mu, self.sigma = self.estimateGaussian(X_train)
        # 选取阈值
        self.epsilon = self.selectThreshold(X_valid, y_valid)
    def estimateGaussian(self, X):
        '''
        @note: 估计高斯分布的参数
        '''
        mu = np.mean(X, axis=0)     # 计算每个维度上的均值
        X_rm_mean = X - mu          # 去均值化
        sigma = sigma = X_rm_mean.T.dot(X_rm_mean) / X.shape[0]
        return mu, sigma
    def selectThreshold(self, X, y_true, n_step=100):
        '''
        @param X: data
        @param y_true: label
        @param y_pred: probability
        @return {float} threshold
        @return {float} f1_score
        @note:  阈值越大, 准确度越高, 召回率越低; 
                这里用均匀步长，可以使用1-exp(x)等指数族函数，在接近中心处时减小步长，远离中心时增加步长
        '''
        # 计算概率
        y_pred_prob = np.zeros(shape=y_true.shape)
        for i in range(y_pred_prob.shape[0]):
            y_pred_prob[i] = self.predict_prob(X[i], self.mu, self.sigma)   # 每个样本的高斯分布概率
        # plt.figure('y_pred_prob'); plt.scatter(X[:, 0], X[:, 1], c=y_true); plt.show()    # 显示概率分布图像
        
        # 选取阈值: 均匀步长
        maxProb, minProb = np.max(y_pred_prob), np.min(y_pred_prob)
        step_size = (maxProb - minProb) / n_step
        thres = np.arange(minProb, maxProb, step_size)

        p = []; r = []; f = []                                  # 保存过程中的score值
        for i in range(thres.shape[0] - 1):
            y_pred_label = self.prob2label(y_pred_prob, thres[i])    # 根据阈值，分割概率，估计标签
            # plt.ion(); plt.figure('y_p_l'); plt.scatter(X[:, 0], X[:, 1], c=y_pred_label); plt.pause(1)
            prec = self.score_precision(y_true, y_pred_label, pos=0)
            rec  = self.score_recall(y_true, y_pred_label, pos=0)
            f1   = self.score_f1(prec, rec)
            if np.isnan(prec) or np.isnan(rec) or np.isnan(f1): break
            p.append(prec); r.append(rec); f.append(f1)
        p = np.array(p); r = np.array(r); f = np.array(f)
        plt.ioff(); plt.figure('score')
        thres = thres[: p.shape[0]]
        plt.plot(thres, p, c='r'); plt.plot(thres, r, c='g'); plt.plot(thres, f, c='b')
        plt.show()
        return thres[np.argmax(f)]
    def prob2label(self, y_pred_prob, threshold):
        '''
        @param y_pred_prob: probability
        @threshold {float}
        @return y_pred_label: label
        '''
        y_pred_label = -np.ones(shape=y_pred_prob.shape, dtype='int8')
        y_pred_label[y_pred_prob >= threshold] = 0  # 大于阈值为正常，即负样本
        y_pred_label[y_pred_prob <  threshold] = 1  # 小于阈值为异常，即正样本
        return y_pred_label
    # ------------------------------------------------------------------------------------------------
    # @param {numpy array} y_true: label
    # @param {numpy array} y_pred: label
    # @return {float}
    def score_precision(self, y_true, y_pred, pos=1):
        '''
        @param pos: 正样本的标签
        @note: precision = TP / (TP + NP); 所有预测为正的样本中，预测正确的比例，正确率
        '''
        neg = 1 - pos
        y_true_pred_pos       = y_true[y_pred==pos]                       # 所有被预测为正的样本
        y_true_pred_pos_true  = y_true_pred_pos[y_true_pred_pos==pos]     # 预测为正的样本中，正确预测的样本
        # y_true_pred_pos_false = y_true_pred_pos[y_true_pred_pos==neg]   # 预测为正的样本中，错误预测的样本
        n_predP  = y_true_pred_pos.shape[0]
        n_predTP = y_true_pred_pos_true.shape[0]
        if n_predP == 0: return np.nan
        return n_predTP / n_predP
    def score_recall(self, y_true, y_pred, pos=1):
        '''
        @note: recall = TP / (TP + FN); 所有实际为正的样本中，预测为正的比例， 查全率
        '''
        neg = 1 - pos
        y_pred_true_pos       = y_pred[y_true==pos]                       # 所有实际为正的样本
        y_pred_true_pos_true  = y_pred_true_pos[y_pred_true_pos==pos]     # 实际为正的样本中，正确预测的样本
        # y_pred_true_pos_false = y_pred_true_pos[y_pred_true_pos==neg]   # 实际为正的样本中，错误预测的样本
        n_trueP  = y_pred_true_pos.shape[0]
        n_trueTP = y_pred_true_pos_true.shape[0]
        if n_trueP == 0: return np.nan
        return n_trueTP / n_trueP
    def score_f1(self, prec, rec):
        '''
        @note: F1 = 2 * P * R / (P + R)
        '''
        if prec==np.nan or rec==np.nan or (prec + rec == 0): return np.nan
        return 2 * prec * rec / (prec + rec)
    # ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    X, Xval, yval = load_data.load_X_X_yval(load_data.ADfile1)

    estimator = AnomalyDetection()
    estimator.fit(X, Xval, yval)

    yval_pred = estimator.predict(X)
    prec = estimator.score_precision(yval, yval_pred, pos=0)
    rec  = estimator.score_recall(yval, yval_pred, pos=0)
    f1 = estimator.score_f1(prec, rec)
    print('validating data')
    print('score of precision is %f, score of recall is %f, score of f1 is %f' % (prec, rec, f1))

    y_pred = estimator.predict(X)
    y_true = np.zeros(shape=y_pred.shape)
    prec = estimator.score_precision(y_true, y_pred, pos=0)
    rec  = estimator.score_recall(y_true, y_pred, pos=0)
    f1 = estimator.score_f1(prec, rec)
    print('training data')
    print('score of precision is %f, score of recall is %f, score of f1 is %f' % (prec, rec, f1))
    pass
    