
"""
	Functions for preprocessing the data, before classification
"""

from numpy import bincount, array, min, zeros, nonzero, where, sum
from settings import NCLASSES

def obtain_class_weights(true_classes, weight_calc = "inverted"):
	"""
	Given the true classes of some samples, gives a vector of sample weights.
	The sample weight is a function of the prior probabity of the class the sample belongs to
	By default, the weight is set to inverted, which means the weight is 1 over the prior probability
	This ensures that every class has the same total weight. For now, this is also the only possibility,
	but more  can be added
	:param true_classes: the true classes of some samples
	"""

	def inverted(priorprob):
		return 1.0 / priorprob

	funcDict = {"inverted" : inverted}

	weightFunc = funcDict[weight_calc]
	priorprobs = bincount(true_classes, minlength = NCLASSES)
	priorprobs = priorprobs / sum(priorprobs).astype(float)
	newprobs =  array([weightFunc(priorprobs[true_classes[i]]) for i in range(len(true_classes))])
	factor = sum(newprobs) / len(newprobs)
	return newprobs / factor

def equalize_class_sizes(data, classes):
	"""
		Equalize classes by removing samples to make them all the same size.

		:param min_size: The number of samples to use for each class.
		:return: trimmmed data and classes.
	"""
	classsizes = bincount(classes)
	print(classsizes)
	min_size = min(classsizes[nonzero(classsizes)]) #ignore classes with 0 entries
	print(min_size)
	filter = zeros(classes.shape, dtype = bool)
	for cls in range(1, NCLASSES + 1):
		this_cls = where(classes == cls)[0][:min_size]
		filter[this_cls] = True
	return data[filter], classes[filter]

if __name__ == '__main__':
	#print undersample(array([[1,2],[2,3],[4,5],[3,2],[3,1],[2,9],[4,6]]), array([1,2,2,1,2,3,4]))
	print "tests later pls"