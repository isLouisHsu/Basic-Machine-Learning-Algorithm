import sklearn.datasets as sd
import numpy as np
import pandas as pd
import operator

def loadData(file):
    data = pd.read_csv(file)
    columns = data.columns.values.tolist()
    if 'Id' in columns:
        del data['Id']
    if 'Score' in columns:
        del data['Score']
    return data

class Node():
    def __init__(self, axis, layer):
        self.axis = axis
        self.layer = layer
        self.nodes = None   # 当不为叶子节点时，类型为字典；叶子节点时即叶子节点的值

class DecisionTree():
    def __init__(self, axis, trainData):
        self.axis = axis    # 叶子节点所在列
        self.trainData = trainData
        self.lenTrain = self.trainData.shape[0]
        self.lenTrainProceed = 0
    # def __init__(self, axis):
    #     self.axis = axis    # 叶子节点所在列
    #     self.trainData = self.loadData('./train.csv')[0:80]  # 'Col_1'至'Col_32'
    #     self.testData = self.loadData('./test.csv')[0:20]
    #     self.lenTrain = self.trainData.shape[0]
    #     self.lenTrainProceed = 0
    #     # ------------------------------------------------------------------------
    # def loadData(self, file):
    #     # 载入数据(dataFrame)，删除不需要的列
    #     data = pd.read_csv(file)
    #     columns = data.columns.values.tolist()
    #     if 'Id' in columns:
    #         del data['Id']
    #     if 'Score' in columns:
    #         del data['Score']
    #     return data
        # ----------------------- 以下是一些计算用的基本函数 -----------------------
    def frequency(self, data, axis):
        # 在输入的data中，以axis标签列进行统计频数。返回字典
        retDict = dict()
        group = data.groupby(axis)  # 以axis列的分类进行分组
        idx = group.count().index   # idx为包含axis列分类的列表
        for i in idx:               # 统计频数
            retDict[i] = data[data[axis]==i].shape[0]/data.shape[0]
        return retDict
    def splitDatasets(self, data, axis):
        # 在输入的data中，以axis标签列进行划分数据
        retDict = dict()
        group = data.groupby(axis)
        idx = group.count().index
        for i in idx:
            retDict[i] = data[data[axis]==i]    # 挑选出axis列中各分类下，其余的数据
            # del retDict[i][axis]                # 删除该列
        return retDict
    def entropy(self, data, axis):
        # 输入的数据中，以axis标签列求得的熵
        ret = 0
        freqDict = self.frequency(data, axis)
        for k, v in freqDict.items():
            ret += -v*np.log(v)
        return ret
    def relativeEntropy(self, data, axis):
        # 计算axis特征下经验条件熵
        ret = 0
        freqDict = self.frequency(data, axis)       # 该特征下的频率统计情况
        splitDict = self.splitDatasets(data, axis)  # 以该特征将数据集划分
        for k, v in splitDict.items():
            ret += freqDict[k]*self.entropy(v, self.axis)    # 条件熵，可理解为各个标签下熵的期望
        return ret
        # ----------------------------- 以下创建树 -------------------------------
    def creatTree(self, data, layer):
        # 以当前输入的data，选择熵减最大的特征建立树分支
        # print(layer)
        bestAxis, b = self.chooseBestAxis(data)
        tree = Node(bestAxis, layer)
        if b==False:
            lable = list(set(data[self.axis]))
            tree.nodes = lable[0]
            self.lenTrainProceed += len(lable)
            print(100*self.lenTrainProceed/self.lenTrain)
            return tree
        tree.nodes = dict()
        dataDict = self.splitDatasets(data, bestAxis)
        # dataDict = self.splitDatasets(self.trainData, bestAxis)
        for k, v in dataDict.items():
            tree.nodes[k] = self.creatTree(v, layer+1)
        return tree
    def chooseBestAxis(self, data):
        # 选择熵减最大的特
        entropyOfData = self.entropy(data, self.axis)
        if entropyOfData==0:    #此时已划分为分类标签
            return self.axis, False
        inforGainDict = dict()
        columns = data.columns.values.tolist()
        for c in columns:
            if c!=self.axis:
                inforGainDict[c] = entropyOfData - self.relativeEntropy(data, c)        # 计算不同特征下信息增益
        inforGainList = sorted(inforGainDict.items(), key=operator.itemgetter(1))    # 按value排序
        return inforGainList[-1][0], True
        # ----------------------------- 打印树 -------------------------------
    def printTree(self, tree, layer):
        print(' '*layer, tree.axis)
        if tree.axis!=self.axis:
            layer += 1
            for k, v in tree.nodes.items():
                self.printTree(v, layer)
        # ----------------------------- 评价树 -------------------------------
    def makeDecision(self, data, node):
        # 对输入的数据(Series)作决策
        if node.axis==self.axis:
            return node.nodes
        else:
            k = data[node.axis]            # 当前节点以axis特征分类
            if k in node.nodes.keys():
                node_new = node.nodes[k]    # 获取字典中以k为键的节点
                return self.makeDecision(data, node_new)
            else:
                 return None
    def evaluateTree(self, data):
        # 计算准确度
        correct = 0
        for i in range(data.shape[0]):
            # print(i/data.shape[0])
            decision = self.makeDecision(data.iloc[i,], DT.tree)
            truth = data.at[i, self.axis]
            if decision==truth:
                correct += 1
        return 100*correct/data.shape[0]

if __name__ == '__main__':
    trainData = loadData('./train.csv')
    testData = loadData('./test.csv')
    DT = DecisionTree('Col_32', trainData[0:100])
    DT.tree = DT.creatTree(DT.trainData, 0)
    print('正确率： ', DT.evaluateTree(trainData[0:100]), '%')
    print('正确率： ', DT.evaluateTree(testData), '%')