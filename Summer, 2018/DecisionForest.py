import DecisionTree as DT

class DecisionForest():
    def __init__(self, data, axis, n):
        self.trainData = data   # 训练数据集
        self.axis = axis        #　划分标签
        self.n = n              # 决策树数目
        self.m = self.trainData.shape[0]//self.n    # 每个决策树训练样本数目
        self.forest = dict()
    def creatForest(self):  
        for n in range(self.n):
            trainData = self.trainData[n*self.m:(n+1)*self.m]
            Tree = DT.DecisionTree(self.axis, trainData)
            Tree.tree = Tree.creatTree(Tree.trainData, 0)
            self.forest[n] = Tree
            print('proceed: ', 100*(n+1)/self.n, '%')
    def makeDecision(self, data):
        # 对单个样本进行预测
        decision = []
        for n in range(self.n):
            Tree = self.forest[n]
            decision.append(Tree.makeDecision(data, Tree.tree))
            print('proceed: ', 100*(n+1)/self.n, '%')
        return max(set(decision), key=decision.count)
    def evaluateForest(self, data):
        correct =0
        for i in range(data.shape[0]):
            decision = self.makeDecision(data.iloc[i,])
            truth = data.at[i, self.axis]
            if decision==truth:
                correct += 1
            print('===========================================')
            print('proceed: ', 100*i/data.shape[0], '%')
        return 100*correct/data.shape[0]

if __name__ == '__main__':
    trainData = DT.loadData('./train.csv')
    testData = DT.loadData('./test.csv')
    DF = DecisionForest(trainData[0:10*500], 'Col_32', 10)
    DF.creatForest()
    print("准确度： ", DF.evaluateForest(trainData[0:10*500]), '%')
    print("准确度： ", DF.evaluateForest(testData), '%')
    pass