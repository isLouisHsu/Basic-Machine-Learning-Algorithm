import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import matplotlib.pyplot as plt
import sklearn.cluster as skclus

class GMM():
    def __init__(self):
        self.X = self.loadData()
        self.n_clusters = 3
        self.n_features = 2
        self.n_samples = 300
        pass
    def loadData(self):
        mu1 = np.array([[6, 6]])
        s1 = np.array([[1, -0.5], [-0.5, 1]])
        X1 = np.dot(np.random.randn(150, 2), nl.cholesky(s1)) + mu1
        mu2 = np.array([[10, 10]])
        s2 = np.array([[2, 0.5], [0.5, 1]])
        X2 = np.dot(np.random.randn(50, 2), nl.cholesky(s2)) + mu2
        mu3 = np.array([[10, 0]])
        s3 = np.array([[2, 0.5], [0.5, 2]])
        X3 = np.dot(np.random.randn(100, 2), nl.cholesky(s3)) + mu3
        X = np.r_[X1, X2, X3]
        return X
    def multiNormal(self, x, mu, sigma):
        # 多元高斯分布
        # x.shape = (n,)
        # mu.shape = (n,)
        n = x.shape[0]
        z = (x-mu).T.dot(nl.inv(sigma)).dot(x-mu)
        res = np.exp(-0.5*z)/np.sqrt(2*np.pi*nl.det(sigma))
        return res
    def gmm(self, delta):
        """
        KMeans只能获取聚类的中心点，实际需求的是一个连续的模型
        EM算法最终返回连续性高斯模型
        将测试数据输入后，可利用softmax等获取后验概率
        """
        # 先进行KMeans聚类
        clf = skclus.KMeans(n_clusters=self.n_clusters)
        clf.fit(self.X)
        # print(clf.cluster_centers_)
        # print(clf.labels_)
        # 计算先验概率P和协方差矩阵Σ
        mu = clf.cluster_centers_           # μ
        P = np.zeros(self.n_clusters)       # Φ
        SM = np.zeros((self.n_clusters, self.n_features, self.n_features))  # Σ
        for k in range(self.n_clusters):
            X = self.X[np.where(clf.labels_==k)]    # 筛选出各类别的样本
            P[k] = X.shape[0]/self.X.shape[0]       # 先验概率
            SM[k] = np.cov(X.T)                     # 协方差矩阵(X两列为两个维度，Cov(xi ,xj))
        
        while True:
            mu_ = mu.copy()    # 保存旧参数，其与新参数之间距离作为停止条件
            # E-step:计算软聚类γ
            gama = np.zeros((self.n_samples, self.n_clusters))
            for i in range(self.n_samples):             # 各个样本
                for k in range(self.n_clusters):        # 各个类别
                    denominator = 0
                    for j in range(self.n_clusters):
                        N = self.multiNormal(self.X[i], mu[j], SM[j])
                        tmp = P[k]*N
                        denominator += tmp
                        if j==k: numerator = tmp
                    gama[i, k] = numerator/denominator
            # M-step:更新数据
            for k in range(self.n_clusters):
                # 计算各个和
                sum1 = 0
                sum2 = 0
                sum3 = 0
                for i in range(self.n_samples):
                    sum1 += gama[i, k]
                    sum2 += gama[i, k]*self.X[i]
                    x_ = np.reshape(self.X[i]-mu[k], (self.n_features, 1))
                    sum3 += gama[i, k]*x_.dot(x_.T)
                P[k]  = sum1/self.n_samples
                mu[k] = sum2/sum1
                SM[k] = sum3/sum1
            # 停止判断
            mu_sum = 0
            for k in range(self.n_clusters):
                mu_sum += nl.norm(mu[k]-mu_[k])
            print(mu_sum)
            if mu_sum < delta: break
        return P, mu, SM
    def predict(self, delta):
        # 使用计算得连续高斯模型，对原有样本集重新划分类别，softmax
        # 判别：P[k]*self.multiNormal(self.X[i], mu[k], SM[k])
        P, mu, SM = self.gmm(delta)
        y_GMM = np.zeros(self.X.shape[0])
        for i in range(self.n_samples):
            post = np.zeros(self.n_clusters)    # 后验概率
            denominator = 0
            for k in range(self.n_clusters):    # 计算分母Σe^pi
                denominator += np.exp(P[k]*self.multiNormal(self.X[i], mu[k], SM[k]))
            for k in range(self.n_clusters):    # 分子e^pi
                post[k] = np.exp(P[k]*self.multiNormal(self.X[i], mu[k], SM[k]))/denominator
            y_GMM[i] = np.argmax(post)

        # 对比KMeans算法，使用KMeans对样本集进行划分
        clf = skclus.KMeans(n_clusters=self.n_clusters)
        clf.fit(self.X)

        # 作图对比
        fx = np.arange(300)
        plt.figure(); plt.axvline(150); plt.axvline(200)
        plt.title("GMM")
        plt.scatter(fx, y_GMM)
        plt.figure(); plt.axvline(150); plt.axvline(200)
        plt.title("KMeans")
        plt.scatter(fx, clf.labels_)
        plt.show()

if __name__ == '__main__':
    model = GMM()
    model.predict(delta=0.0001)

    # 实际参数
    # mu1 = np.array([[6, 6]])
    # s1 = np.array([[1, -0.5], [-0.5, 1]])
    # X1 = np.dot(np.random.randn(150, 2), nl.cholesky(s1)) + mu1
    # mu2 = np.array([[10, 10]])
    # s2 = np.array([[2, 0.5], [0.5, 1]])
    # X2 = np.dot(np.random.randn(50, 2), nl.cholesky(s2)) + mu2
    # mu3 = np.array([[10, 0]])
    # s3 = np.array([[2, 0.5], [0.5, 2]])
    # X3 = np.dot(np.random.randn(100, 2), nl.cholesky(s3)) + mu3
