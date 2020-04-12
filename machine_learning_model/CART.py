import numpy as np

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feat_idx, value):
    lSet = dataSet[dataSet[:, feat_idx] <= value]
    rSet = dataSet[dataSet[:, feat_idx] > value]
    return lSet, rSet


def reg_leafValue(dataSet):  #计算数据集中，label的均值
    return np.mean(dataSet[:,-1])


def reg_dataSetErr(dataSet):  #计算数据集中，标签的均方差之和
    mean = np.mean(dataSet[:, -1])
    sum_MSE = 0
    for e in dataSet[:, -1]:
        sum_MSE += (e - mean) ** 2
    return sum_MSE


def chooseBestSplit(dataSet, leafValue=reg_leafValue, dataSetErr=reg_dataSetErr, ops=(1, 4)):
    err_redu = ops[0]; n_min_exam = ops[1]

    if len(set(dataSet[:, -1])) == 1:  # 如果数据集的标签都一样，则没必要切分，退出
        return None, leafValue(dataSet)

    n_row, n_col = np.shape(dataSet)
    err = dataSetErr(dataSet)
    min_err = np.inf; best_feat_idx = 0; bestValue = 0
    for feat_idx in range(n_col - 1):  # 遍历所有特征f
        for splitVal in set(dataSet[:, feat_idx]):  # 遍历所有取值v
            lSet, rSet = binSplitDataSet(dataSet, feat_idx, splitVal)  # 根据(f,v)划分数据集
            if (np.shape(lSet)[0] < n_min_exam) or (np.shape(rSet)[0] < n_min_exam):
                continue
            new_err = dataSetErr(lSet) + dataSetErr(rSet)  # 然后计算这两份数据集的均方差之和
            if new_err < min_err:  # 选择均方差之和最小的(f,v)
                best_feat_idx = feat_idx
                bestValue = splitVal
                min_err = new_err

    if (err - min_err) < err_redu:  # 如果切分后，“误差的下降程度”小于阈值，停止切分
        return None, leafValue(dataSet)

    subSet0, subSet1 = binSplitDataSet(dataSet, best_feat_idx, bestValue)
    if (np.shape(subSet0)[0] < n_min_exam) or (np.shape(subSet1)[0] < n_min_exam):
        return None, leafValue(dataSet)  # 如果切分后的数据集，样本过少，则退出

    return best_feat_idx, bestValue  # 返回最佳特征，及对应的取值


def createTree(dataSet, leafValue=reg_leafValue, dataSetErr=reg_dataSetErr, ops=(1,4)):
    feat_idx, val = chooseBestSplit(dataSet, leafValue, dataSetErr, ops)  #根据“特征选取准则”，选择最佳（特征，值）
    if feat_idx == None:  #如果满足停止条件，则返回叶节点的取值
        return val
    regTree = {}
    regTree['spIdx'] = feat_idx  #根据最佳（feat，value），建立节点
    regTree['spVal'] = val

    # 根据最佳（feat，value），切分数据集，
    # 连续特征：<=value为左子集，>value为右子集；类别特征：=value为左子集，否则为右子集
    lSet, rSet = binSplitDataSet(dataSet, feat_idx, val)

    # 每个分支的子数据集都按同样的方式继续划分；直到满足停止条件，输出叶节点取值
    regTree['left'] = createTree(lSet, leafValue, dataSetErr, ops)
    regTree['right'] = createTree(rSet, leafValue, dataSetErr, ops)

    return regTree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0:  # 如果某个“非叶节点”没有验证集，剪枝
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):  # 将验证集的数据，一步步分割，“分摊”到每个节点上
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    # 如果找到叶节点的父节点（该节点的左右分支都是叶节点），看看是否需要把合并这两个叶节点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(np.power(rSet[:, -1] - tree['right'], 2))  #计算未合并的损失
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))  #计算合并后的损失
        if errorMerge < errorNoMerge:  #如果合并后的损失 < 未合并的损失；则合并
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def regression(regTree, testVec):
    spIdx = regTree['spIdx']
    spVal = regTree['spVal']

    if testVec[spIdx] <= spVal:
        if type(regTree['left']).__name__ == 'dict':
            value = regression(regTree['left'], testVec)
        else:
            value = regTree['left']
    else:
        if type(regTree['right']).__name__ == 'dict':
            value = regression(regTree['right'], testVec)
        else:
            value = regTree['right']

    return value


if __name__ == '__main__':
    #注意，数据必须是np.array的形式
    train_data = np.array(loadDataSet('ex0.txt'))
    myTree = createTree(train_data)
    print(myTree)
    print(regression(myTree, [0.1, 0.1]))



    {'spIdx': 1,
     'spVal': 0.39435,
     'left': {'spIdx': 1,
              'spVal': 0.197834,
              'left': -0.023838155555555553,
              'right': 1.0289583666666666},
     'right': {'spIdx': 1,
               'spVal': 0.582002,
               'left': 1.980035071428571,
               'right': {'spIdx': 1,
                         'spVal': 0.797583,
                         'left': 2.9836209534883724,
                         'right': 3.9871632}}}
