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
    def __init__(self, index, leftNode=None, rightNode=None):
        self.index = index          # 分类特征
        self.leftNode = leftNode    # X[:, idx]==0样本生成的左结点
        self.rightNode = rightNode  # X[:, idx]==1样本生成的右结点

class DecisionTree():
    def __init__(self):
        self.inputEncoder = OneHotEncoder() # 这里的输入各特征都是categorical型
        self.usedFeature = []
        self.tree = None
    def fit(self, X, y):
        X_encoded = self.inputEncoder.fit_transform(X).toarray()
        y_encoded = OneHotEncoder().fit_transform(y.reshape((-1, 1))).toarray()
        self.tree = self.creatNode(X_encoded, y_encoded)
    def creatNode(self, X_encoded, y_encoded):
        # 统计维度数目
        n_feature, n_label = X_encoded.shape[1], y_encoded.shape[1]
        # 判断是否全部属于同一类
        y_count = np.sum(y_encoded, axis=0)                             # 统计属于各类别的样本数目
        idx_nonzero = np.array(np.nonzero(y_count)).reshape((-1,))      # 统计不为0的列索引
        if idx_nonzero.shape[0] == 1: return int(idx_nonzero[0])        # 如果只有一列不为0，则表示输入的样本全都属于同一类; 若不转换为int，其类型为np.int64，难以判断类型
        # 计算经验熵 - H(D) = -Σplogp
        p_y_encoded = np.mean(y_encoded, axis=0)                
        H_D = - np.sum(p_y_encoded * np.log(p_y_encoded))
        # 计算经验条件熵 - H(D|A) = -Σ(pi*H(Di))
        H_D_A = np.zeros(shape=(n_feature,))                            # 统计各维特征的经验条件熵
        for j in range(n_feature):                                      # 各个特征
            a = X_encoded[:, j] # for debug, to be deleted later
            if j in self.usedFeature: continue                            # 若该列特征已使用，不再继续计算
            p_X_feature_j = np.mean(X_encoded[:, j])
            for v in range(2):                                          # 特征二元取值 v∈{0, 1}
                idx_selected = (X_encoded[:, j]==v)             
                y_selected = y_encoded[idx_selected]                    # 筛选出特征j取值为v的数据
                if y_selected.shape[0]==0: continue                     # 取值只有一种的特征列，则筛选另一种取值时为空，无需计算
                p_y_selected = np.mean(y_selected, axis=0)              # 在特征j在取值v情况下，y的概率分布
                # H_D_v = - np.sum(p_y_selected * np.log(p_y_selected)) # ！注意这样写有问题，因为 0*np.log(0)=nan
                H_D_v = 0
                for l in range(n_label): 
                    H_D_v -= 0 if p_y_selected[l]==0\
                        else p_y_selected[l] * np.log(p_y_selected[l])  # 特征j取值为v时的经验熵
                # H_D_v的计算用以上代替
                H_D_A[j] += (p_X_feature_j if v==1 else (1 - p_X_feature_j)) * H_D_v    # 特征j的经验条件熵，乘特征j的概率分布可理解成期望
        # 计算相对熵/信息增益 - g(D, A) = H(D) - H(D|A)
        g_D_A = H_D - H_D_A 
        # 选出相对熵/信息增益最大的特征作为分支
        idx_j = int(np.argmax(g_D_A))       # 若不转换为int，其类型为np.int64，难以判断类型
        # 该列不再用于分类
        self.usedFeature.append(idx_j)
        # 以选中的特征的取值作为分支判别条件，继续生成结点
        idx_j_left, idx_j_right = (X_encoded[:, idx_j]==0), (X_encoded[:, idx_j]==1)
        # 选出数据，用于左右分支继续生成结点
        X_left,  y_left  = X_encoded[idx_j_left],  y_encoded[idx_j_left]
        X_right, y_right = X_encoded[idx_j_right], y_encoded[idx_j_right]
        leftNode  = self.creatNode(X_left,  y_left)
        rightNode = self.creatNode(X_right, y_right)
        return Node(idx_j, leftNode, rightNode)
    def predict(self, X):
        X_encoded = self.inputEncoder.transform(X).toarray()
        y_pred = np.ones(shape=(X_encoded.shape[0],))
        for i in range(y_pred.shape[0]):
            nextNode = self.tree.leftNode if X_encoded[i][self.tree.index]==0 else self.tree.rightNode
            while not isinstance(nextNode, int):    # 下一个结点保存的数据类型不为整数(即为Node)
                nextNode = nextNode.leftNode if X_encoded[i][nextNode.index]==0 else nextNode.rightNode
            y_pred[i] = nextNode
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
    X = X.astype('int')
    # encode the label
    for r in range(y.shape[0]):
        y[r] = dict_encode[4][y[r]]
    y = y.astype('int')
    # train the estimator
    estimator = DecisionTree()
    estimator.fit(X, y)
    # predict the output of training data, calculate the accuracy score
    y_pred = estimator.predict(X)
    print(estimator.score(y, y_pred))