import numpy as np

X = [
    [1, 'S'],
    [1, 'M'],
    [1, 'M'],
    [1, 'S'],
    [1, 'S'],
    [2, 'S'],
    [2, 'M'],
    [2, 'M'],
    [2, 'L'],
    [2, 'L'],
    [3, 'L'],
    [3, 'M'],
    [3, 'M'],
    [3, 'L'],
    [3, 'L']
]
y = [-1 ,-1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
X = np.array(X); y = np.array(y)
X_test = np.array([2, 'S'])
'''
            P(yc)ΠP(Xi|yc)
P(yc|X) = ——————————————————
                ΠP(Xi)
'''
X_cate = [['1', '2', '3'], ['S', 'M', 'L']]
Y_cate = [-1, 1]

p_yc = np.zeros(shape=(2,))
for i in range(2):
    p_yc[i] = y[y==Y_cate[i]].shape[0] / y.shape[0]

p_Xi_yc = np.zeros(shape=(2, 2, 3))     # n_class x n_feature x n_value
for i in range(2):
    X_yi = X[y==Y_cate[i]]              # 筛选出y = yi的数据
    for j in range(2):
        X_yi_Xj = X_yi[:, j]
        for k in range(3):
            val = X_cate[j][k]
            p_Xi_yc[i, j, k] = X_yi_Xj[X_yi_Xj==val].shape[0] / X_yi_Xj.shape[0]

p_Xi = np.zeros(shape=(2, 3))          # n_feature x n_value
for i in range(2):
    Xi = X[:, i]
    for j in range(3):
        p_Xi[i, j] = Xi[Xi==X_cate[i][j]].shape[0] / Xi.shape[0]

p_yc_Xi = np.zeros(shape=(2,)) 
for i in range(2):
    p_yc_Xi[i] = p_yc[i]
    for j in range(2):
        p_yc_Xi[i] *= p_Xi_yc[i, j, X_cate[j].index(X_test[j])]

print(Y_cate[np.argmax(p_yc_Xi)])
            