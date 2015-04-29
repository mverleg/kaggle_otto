
from numpy import logical_not, concatenate, copy
from numpy.random import permutation
import numpy as np

def equalize_train_classes(max_samples, class_set, class_set_labels):
	numClasses = 9
	residualData = np.empty((0,93))
	residualLabels = np.empty((0))
	trainData = np.empty((0,93))
	trainLabels = np.empty((0))

	for clsi in range(0, numClasses):
		nr_samples = min(max_samples, class_set[clsi].shape[0])
		# print class_set[clsi].shape[0], max_samples
		index = permutation(class_set[clsi].shape[0])[:nr_samples]
		resIndex = list(set(permutation(class_set[clsi].shape[0])) - set(index))
		# print len(index), len(resIndex)

		trainData = np.vstack((trainData, class_set[clsi][index, :]))
		trainLabels = np.append(trainLabels, class_set_labels[clsi][index])
		if len(resIndex)>0:
			residualData = np.vstack((residualData, class_set[clsi][resIndex, :]))
			residualLabels = np.append(residualLabels, class_set_labels[clsi][resIndex])

	return trainData, trainLabels, residualData, residualLabels


