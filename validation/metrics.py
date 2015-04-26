
from numpy import bincount, float64, clip
from sklearn.metrics import confusion_matrix as calc_confusion_matrix
from settings import NCLASSES
from validation.score import prob_to_cls


def confusion_matrix(predictions, classes_true):
	"""
		Generate the normalized confusion matrix.

		http://stackoverflow.com/questions/20927368/python-how-to-normalize-a-confusion-matrix
	"""
	RCM = calc_confusion_matrix(classes_true, prob_to_cls(predictions), labels = range(1, 10))
	return (RCM.T / float64(RCM.sum(1).clip(min = 1))).T


def average_size_mismatch(predictions, classes_true):
	"""
		Calculate how big each calculated class is as a fraction of the true data size.
	"""
	return bincount(prob_to_cls(predictions), minlength = NCLASSES + 1)[1:] / \
		float64(bincount(classes_true, minlength = NCLASSES + 1)[1:].clip(min = 1))


