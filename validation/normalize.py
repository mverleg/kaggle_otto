
from numpy.random import uniform


def normalize_data(data):
	"""
		Scale each COLUMN such that the training DATA is in the range [0-1]. No guarantees for testing data, since all should use the same scaling factors.
	"""
	#todo


def normalize_probabilities(predictions):
	"""
		Scale each ROW such that the SUM is exactly 1.
	"""
	return predictions / predictions.sum(axis = 1)[:,None]


def rows_normalized(predictions, epsilon = 1e-8):
	"""
		:return: True if rows add up to 1.
	"""
	return all(abs(predictions.sum(axis = 1) - 1) < epsilon)


predictions = uniform(size = (21, 9))
