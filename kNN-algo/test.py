import kNN
import matplotlib.pyplot as plt
from numpy import array

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
print('DATING MATRIX: ', datingDataMat)
print('DATING LABELS: ', datingLabels[0:20])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
           15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()
