import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sd
import sklearn.svm as ss

class SVM():
    def __init__(self, n_samples, C):
        self.m = n_samples
        self.C = C
        self.X, self.Y = self.creatData(self.m)
    def creatData(self, n_samples):
        #关键参数有n_samples（生成样本数）， n_features（样本特征数）， n_redundant（冗余特征数）和n_classes（输出的类别数）
        # X1为样本特征，Y1为样本类别输出， 共400个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
        X, Y = sd.make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_classes=2, n_clusters_per_class=1)
        for i in range(self.m):
            Y[i] = 1 if Y[i]==1 else -1
        return X, Y
    def showFigure(self, W, b):
        x1 = self.X[:, 0]
        x2 = self.X[:, 1]
        plt.figure()
        plt.scatter(x1, x2, c=self.Y)
        X1 = np.linspace(min(x1), max(x1))
        X2 = -(W[0]*X1+b)/W[1]   # self.W[0]*x1 + self.W[1]*x2 + b = 0
        # plt.scatter(X1, X2)
        plt.plot(X1, X2)
        plt.show()
    # -------------------------------------------------
    def randomChooseJ(self, i):
        # 随机选择与i不同的j
        j = i
        while(i==j):
            j = np.random.randint(0, self.m)
        return j
    # -------------------------------------------------
    def K(self, i, j):
        # 求Xi与Xj的内积
        # 在此处可修改核函数？
        return np.dot(self.X[i,:], self.X[j,:].T)
    def f(self, i, lda, b):
        # 求fi
        ret = 0
        for m in range(self.m):
            ret += lda[m]*self.Y[m]*self.K(i, m)
        return ret + b
    def E(self, i, lda, b):
        # 求偏差Ei
        return self.f(i, lda, b) - self.Y[i]
    def clip(self, num, L, H):
        # 限定范围
        if num>H:
            return H
        elif num<L:
            return L
        else:
            return num
    def getW(self, lda):
        ret = 0
        for i in range(self.m):
            ret += lda[i]*self.Y[i]*np.reshape(self.X[i], (2, 1))
        return ret
    # -------------------------------------------------
    def smoDemo(self, maxIter, delta):
        # 参数初始化
        lda = np.ones(shape=(self.m, 1))
        b = 0
        nIter = 0
        while(nIter<maxIter):
            updated = 0
            for i in range(self.m):
                Ei = self.E(i, lda, b)          # 计算Ei
                if (Ei*self.Y[i]<-delta and lda[i]<self.C) or (Ei*self.Y[i]>delta and lda[i]>0):
                    j = self.randomChooseJ(i)   # 这里只是随机选择，没有选择|Ei-Ej|最大的Xj
                    Ej = self.E(j, lda, b)      # 计算Ej
                    ldai = lda[i, 0]            # 保存更新前参数
                    ldaj = lda[j, 0]            # 保存更新前参数
                    Kii = self.K(i, i)
                    Kij = self.K(i, j)
                    Kjj = self.K(j, j)
                    # 计算lda[j]的上下界
                    if self.Y[i]==self.Y[j]:
                        L = max(0, lda[i]+lda[j]-self.C)
                        H = min(self.C, lda[i]+lda[j])
                    else:
                        L = max(0, lda[j]-lda[i])
                        H = min(self.C, lda[j]-lda[i]+self.C)
                    if L==H: 
                        # print('L==H')
                        continue
                    # 求二次项系数eta
                    eta = Kii + Kjj - 2.0*Kij
                    if eta <= 0: 
                        # print('eta <= 0')
                        continue # 实际应讨论二次项系数的符号，不同情况下进行不同的计算
                    # 更新lda[j]
                    lda[j] += self.Y[j]*(Ei - Ej)/eta
                    # 修剪lda[j]
                    lda[j] = self.clip(lda[j], L, H)
                    if abs(ldaj-lda[j])<0.0001: 
                        # print('lda[j]变化太小')
                        continue
                    # 更新lda[i]
                    lda[i] += self.Y[i]*self.Y[j]*(ldaj-lda[j]) 
                    # 更新b
                    bi = b - Ei + (ldai-lda[i])*self.Y[i]*Kii + (ldaj-lda[j])*self.Y[j]*Kij
                    bj = b - Ej + (ldai-lda[i])*self.Y[i]*Kij + (ldaj-lda[j])*self.Y[j]*Kjj
                    # 修剪b
                    if lda[i]>0 and lda[i]<self.C:   b = bi
                    elif lda[j]>0 and lda[j]<self.C: b = bj
                    else:                            b = 0.5*(bi+bj)
                    updated += 1
                    print("第%d次迭代时样本%d,%d进行了%d次优化" % (nIter, i, j, updated))
            if updated==0:  # 若lda未更新，表示该lda参数不用更新
                nIter += 1
            else:           # 若lda更新，表示该lda参数仍需要更新，重新迭代
                nIter = 0
        return self.getW(lda), b
    def sklearnSVM(self):
        clf = ss.SVC()
        clf.fit(self.X, self.Y)
        print('point (1,1) belongs to class',clf.predict(np.array([[1,1]])))

if __name__=='__main__':
    # model = SVM(50, C=float('inf'))   # 无解
    model = SVM(50, C=0.6)
    W, b = model.smoDemo(50, 0.001)
    model.showFigure(W, b)
    model.sklearnSVM()