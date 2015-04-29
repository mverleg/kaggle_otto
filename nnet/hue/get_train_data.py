
from numpy import logical_not, concatenate, copy
from numpy.random import permutation


def equalize_train_classes(max_samples, class_set, class_set_labels):
	numClasses = 9
	trainData= []
	trainLabels = []
	residualData = []
	residualLabels = []

	for clsi in range(0, numClasses):
		#assert max_samples <= class_set.shape[0], 'not enough samples'
		nr_samples = min(max_samples, class_set[clsi].shape[0] - 1)
		#print nr_samples, max_samples
		index = permutation(range(class_set[clsi].shape[0]))[:nr_samples]
		#print class_set[clsi].shape, class_set_labels[clsi].shape, index.shape
		#print class_set[clsi].shape
		#print index
		#print len(index)
		trainData.append(copy(class_set[clsi][index, :]))
		#concatenate((trainData, class_set[clsi][index, :]), axis = 0)
		trainLabels.append(copy(class_set_labels[clsi][index]))
		residualData.append(copy(class_set[clsi][logical_not(index), :]))
		residualLabels.append(copy(class_set_labels[clsi][logical_not(index)]))
		#print clsi + 1, len(index), class_set[clsi].shape

	# combine a list of matrices into big matrix
	#print 'res', [r.shape for r in trainData]
	trainData = concatenate(trainData, 0)
	trainLabels = concatenate(trainLabels, 0)
	#print 'res', [r.shape for r in residualData]
	residualData = concatenate(residualData, 0)
	residualLabels = concatenate(residualLabels, 0)

	return trainData, trainLabels, residualData, residualLabels


