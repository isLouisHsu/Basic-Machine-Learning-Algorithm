'''
采用onehot编码的形式进行计算，极度容易过拟合
'''
import numpy as np
from sklearn.preprocessing import OneHotEncoder

dict_encode = [
    {'青年': 0, '中年': 1, '老年': 2},
    {'否':   0, '是': 1},
    {'否':   0, '是': 1},
    {'一般': 0, '好': 1, '非常好': 2},
    {'否':   0, '是': 1}
]
X = [
    ['青年', '否', '否', '一般' ],
    ['青年', '否', '否', '好'   ],
    ['青年', '是', '否', '好'   ],
    ['青年', '是', '是', '一般' ],
    ['青年', '否', '否', '一般' ],
    ['中年', '否', '否', '一般' ],
    ['中年', '否', '否', '好'   ],
    ['中年', '是', '是', '好'   ],
    ['中年', '否', '是', '非常好'],
    ['中年', '否', '是', '非常好'],
    ['老年', '否', '是', '非常好'],
    ['老年', '否', '是', '好'   ],
    ['老年', '是', '否', '好'   ],
    ['老年', '是', '否', '非常好'],
    ['老年', '否', '否', '一般'],
]
y = ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']

class Node():
    def __init__(self):
        self.index = None           # 子树分类标签, 若为叶节点, 则为None
        self.childNode = None       # 子树，若为叶节点, 则为分类标签; 否则为字典

class DecisionTree():
    '''
    @note:  假定输入的特征都是categorical型;
            ID3
    '''
    def __init__(self):
        self.tree = None
    def fit(self, X, y):
        self.tree = self.creatNode(X, y)
        pass
    def creatNode(self, X, y):
        node = Node()
        # 判断是否属于同一类
        if len(set(y)) == 1: node.childNode = list(set(y))[0]; return node
        # 计算经验熵 - H(D)
        y_encoded = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        p_y = np.mean(y_encoded, axis=0)
        p_y[p_y==0.0] = 1.0  # 因为 0*np.log(0)结果为nan, 而不是0, 用 1*np.log(1)替代
        H_D = - np.sum(p_y * np.log(p_y))
        # 计算以每个特征作为分类依据的经验条件熵 - H(D|A)
        H_D_A = np.zeros(shape=(X.shape[1],))                       # 初始化数组
        for i_feature in range(X.shape[1]):                         # 每一个特征
            X_feature = X[:, i_feature]                             # 取出该列特征
            if len(set(X_feature)) == 1: H_D_A[i_feature] = float('inf'); continue
            X_feature_encoded = OneHotEncoder().fit_transform(X_feature.reshape((-1, 1))).toarray()
            p_X = np.mean(X_feature_encoded, axis=0)                # 每个取值的概率
            for j_feature in range(X_feature_encoded.shape[1]):     # 该特征取值有几种，编码后就有几列
                y_encoded_feature = y_encoded[X_feature_encoded[:, j_feature]==1]   # 该特征某种取值下，其对应的标签值
                p_y_X = np.mean(y_encoded_feature, axis=0)
                p_y_X[p_y_X==0.0] = 1.0
                H_D_feature = - np.sum(p_y_X * np.log(p_y_X))
                H_D_A[i_feature] += p_X[j_feature] * H_D_feature    # 条件熵
        # 计算相对熵/信息增益 g(D, A) = H(D) - H(D|A)
        g_D_A = H_D - H_D_A
        # 选出最大的作为分类特征
        node.index = np.argmax(g_D_A)
        X_selected = X[:, node.index]
        # 分类后继续建立树
        node.childNode = dict()
        for val in set(X_selected):
            valIndex = (X_selected==val)
            X_val, y_val = X[valIndex], y[valIndex]
            node.childNode[val] = self.creatNode(X_val, y_val)      # 存储在字典中，键为分类值，值为子树
        return node
    def predict(self, X):
        y_pred = np.zeros(shape=(X.shape[0],))
        for i_sample in range(X.shape[0]):
            currentNode = self.tree                         # 初始化为父节点
            while not currentNode.index==None:              # 若为None, 表示为叶子结点
                val = X[i_sample, currentNode.index]        # 当前样本在分类特征上的值
                currentNode = currentNode.childNode[val]    # 递归
            else:
                y_pred[i_sample] = currentNode.childNode
        return y_pred
    def score(self, y_true, y_pred):
        ''' accuracy '''
        return np.mean(np.equal(y_true, y_pred).astype('float'))

if __name__ == '__main__':
    X = np.array(X); y = np.array(y)
    # encode the data
    for c in range(X.shape[1]):
        for r in range(X.shape[0]):
            X[r, c] = dict_encode[c][X[r, c]]
    X = X.astype('int') # (15, 4)
    # encode the label
    for r in range(y.shape[0]):
        y[r] = dict_encode[4][y[r]]
    y = y.astype('int') # (15, )

    # train the estimator
    estimator = DecisionTree()
    estimator.fit(X, y)
    # predict the output of training data, calculate the accuracy score
    y_pred = estimator.predict(X)
    print(estimator.score(y, y_pred))