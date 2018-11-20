import operator
from numpy import *
from os import listdir


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    # 计算距离
    dataSetSize = dataSet.shape[0]  # shape 矩阵的维数, shape[0]就是矩阵的行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 形成纵向的许多inX,然后和原始的训练样本求差
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()  # 从小到大的下标
    classCount = {}
    for i in range(k):  # 对于取的k个数
        voteIlabel = labels[sortedDistIndicies[i]]  # 对应的label
        classCount[voteIlabel] = classCount.get(
            voteIlabel, 0) + 1  # 给classCount赋值，并且计算每个label的出现次数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(
        1), reverse=True)  # 通过classCount.iteritems()里每一个的第二个维度排序
    return sortedClassCount[0][0]


''' 
读取文件并且转换为python支持的格式输出
'''


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


'''
归一转化
'''


def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 每一列的最小值
    maxVals = dataSet.max(0)  # 每一列的最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


'''
测试代码
'''


def datingClassTest():
    hoRatio = 0.10  # 因为测试数据取十分之一，所以是0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],  # 此处normMat[numTestVecs:m, :]表示另外那90%的样本矩阵
                                     datingLabels[numTestVecs:m], 3)  # 此处normMat[numTestVecs:m, 3] 90%的样本矩阵的第四列，也就是标签
        print "the classifier came back with: %d, the real answer is: %d" \
            % (classifierResult, datingLabels[i])

        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))


'''
预测函数
'''


def classifyPerson():
    resultList = ['not at all', 'in samll doses', 'in large doses']
    percentTats = float(raw_input(
        "percentage of time spent playing video game?"))
    ffMiles = float(raw_input("frequent flier miles earned per year"))
    iceCream = float(raw_input("liters of ice cream consumed per year"))
    datingDataMat, datingLabels = file2matrix('datingTsetSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0(
        (inArr-minVals)/ranges, normMat, datingLabels, 3)
    print "You will probably like this person: ", \
        resultList[classifierResult - 1]


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLines = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,
                                     trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" \
            % (classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
