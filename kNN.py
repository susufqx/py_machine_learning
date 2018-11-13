from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    # 计算距离
    dataSetSize = dataSet.shape[0]  # shape 矩阵的维数, shape[0]就是矩阵的行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 形成纵向的许多inX,然后和原始的训练样本求差
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort() #从小到大的下标
    classCount={}
    for i in range(k): # 对于取的k个数
        voteIlabel = labels[sortedDistIndicies[i]] # 对应的label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 给classCount赋值，并且计算每个label的出现次数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 通过classCount.iteritems()里每一个的第二个维度排序
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()  # make the file as an array
    numberOfLines = len(arrayOLines)  # get lines of the file
    returnMat = zeros((numberOfLines, 3))  # creat a matrix numberOfLines x 3
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 去除首尾空格
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
