
from numpy import vstack, logical_not
from numpy.random import permutation


def equalize_train_classes(max_samples, class_set, class_set_labels):
	numClasses =9
	trainData= []
	trainLabels = []
	residualData = []
	residualLabels = []

	for clsi in range(0, numClasses):
		assert max_samples <= class_set.shape[0], 'not enough samples'
		index = permutation(range(class_set[clsi].shape[0]))[:max_samples]
		trainData.append(class_set[clsi][index, :])
		trainLabels.append(class_set_labels[clsi][index, :])
		residualData.append(class_set[clsi][logical_not(index), :])
		residualLabels.append(class_set_labels[clsi][logical_not(index), :])

	# combine a list of matrices into big matrix
	trainData = vstack(trainData)
	trainLabels = vstack(trainLabels)
	residualData = vstack(residualData)
	residualLabels = vstack(residualLabels)

	return trainData, trainLabels, residualData, residualLabels


