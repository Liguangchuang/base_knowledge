# coding=utf-8
import operator
from math import log
import numpy as np


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return max(classCount)


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def cal_infoGain(dataSet, feat_idx):
    def cal_Ent(dataSet):  #计算熵
        labelCounts = {}
        for feaVec in dataSet:
            currentLabel = feaVec[-1]
            if currentLabel not in labelCounts:
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / len(dataSet)
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt


    def cal_CondiEnt(dataSet, feat_idx):  # 计算条件熵
        n_sample = len(dataSet); condiEnt = 0
        featCol = [example[feat_idx] for example in dataSet]

        for val in set(featCol):
            subDataSet = splitDataSet(dataSet, feat_idx, val)
            n_subSample = len(subDataSet)
            prob = n_subSample / n_sample
            subEnt = cal_Ent(subDataSet)
            condiEnt += prob * subEnt

        return condiEnt

    return cal_Ent(dataSet) - cal_CondiEnt(dataSet, feat_idx)  #信息增益


def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1; bestInfoGain = 0.0; bestFeature = -1

    for feat_idx in range(numFeatures):
        infoGain = cal_infoGain(dataSet, feat_idx)
        if infoGain > bestInfoGain:  #选择“信息增益”最大的特征
            bestInfoGain = infoGain
            bestFeature = feat_idx
    return bestFeature, bestInfoGain


def createTree(dataSet, featNames, min_gain):
    classList = [example[-1] for example in dataSet]
    if len(set(classList)) == 1:  #如果类别相同则停止划分，并输出类别标记
        return classList[0]
    if len(dataSet[0]) == 1:  #如果所有特征已经用完，输出最多的类别
        return majorityCnt(classList)

    bestFeat_idx, bestInfoGain = chooseBestFeature(dataSet)  #计算信息增益，并选出使“信息增益”最大的特征
    bestFeat = featNames[bestFeat_idx]

    if bestInfoGain < min_gain:  #如果信息增益小于阈值，则输出最多的类别
        return majorityCnt(classList)

    myTree = {}
    myTree['bestFeat'] = bestFeat  #根据最佳特征，建立节点

    bestFeat_ValuesSet = set([example[bestFeat_idx] for example in dataSet])
    for value in bestFeat_ValuesSet:
        sub_dataset = splitDataSet(dataSet, bestFeat_idx, value)   #对于最佳特征的每种取值，划分出子数据集
        sub_featNames = featNames[0: bestFeat_idx] + featNames[bestFeat_idx+1: ]

        # 每个取值生成一个分支；并且分支下面的“子数据集”继续按同样的方式划分；直到满足停止条件，输出类别标签
        myTree[value] = createTree(sub_dataset, sub_featNames, min_gain=0.1)
    return myTree




def classify(inputTree, featNames, testVec):
    bestFeat = inputTree['bestFeat']
    bestFeat_idx = featNames.index(bestFeat)

    for key in list(inputTree.keys())[1:]:
        if testVec[bestFeat_idx] == key:
            if type(inputTree[key]).__name__ == 'dict':
                classLabel = classify(inputTree[key], featNames, testVec)
            else:
                classLabel = inputTree[key]
    return classLabel


class DecisionTree():
    def __init__(self, featNames):
        self.featNames = featNames

    def fit(self, X_train, y_train):
        dataSet = [X + [y] for (X, y) in zip(X_train, y_train)]
        self.myTree = createTree(dataSet, self.featNames, min_gain=0.1)
        print(self.myTree)

    def predict(self, X):
        Y_pred = [0] * len(X)

        for row_idx in range(0, len(X)):
            y_label = classify(self.myTree, self.featNames, X[row_idx])
            Y_pred[row_idx] = y_label

        return Y_pred


if __name__ == '__main__':
    #注意，数据必须是list的形式，全部都是基于list实现的

    featNames = ['A1', 'A2']
    train_data = [[1, 1, 'yes'],
                  [1, 1, 'yes'],
                  [1, 0, 'no'],
                  [0, 1, 'no'],
                  [0, 1, 'no']]
    X_train = [example[0: -1] for example in train_data]
    y_train = [example[-1] for example in train_data]
    X_test = X_train

    DT = DecisionTree(featNames)
    DT.fit(X_train, y_train)
    y_pred = DT.predict(X_test)
    print(y_pred)


    {'bestFeat': 'A1',
     0: 'no',
     1: {'bestFeat': 'A2',
         0: 'no', 1:
             'yes'}}
