
"""
	This file contains functions for two normalization operations:

	1. normalize the data, making sure all values are [0, 1]
		normalize_data(..)
	2. normalize calculated probabilities, making sure all rows add up to 1.
		normalize_probabilities(..)
"""

from numpy import abs, float32, array, save, load, float64
from numpy.random import uniform
from utils.loading import get_training_data, get_testing_data
from settings import COLUMN_MAX_PATH, VERBOSITY


def store_column_maxima():
	"""
		Calculate and store the maxima of the combined train and test columns (for normalization).

		This loads all data so it's not fast.
	"""
	train_data = get_training_data()[0]
	test_data = get_testing_data()[0]
	maxima = array([max(tr, te) for tr, te in zip(train_data.max(0), test_data.max(0))])
	save(COLUMN_MAX_PATH, maxima)
	if VERBOSITY >= 1:
		print 'updated the column maxima file'
	return maxima


def get_column_maxima():
	"""
		Load the column maxima. First try memory cache, then file cache, then calculate.
	"""
	try:
		return get_column_maxima._MAXIMA_CACHE
	except AttributeError:
		try:
			get_column_maxima._MAXIMA_CACHE = load(COLUMN_MAX_PATH)
		except:
			get_column_maxima._MAXIMA_CACHE = store_column_maxima()
	print get_column_maxima._MAXIMA_CACHE
	return get_column_maxima._MAXIMA_CACHE.astype(float32)


def normalize_data(data):
	"""
		Scale each COLUMN such that the training and test DATA is in the range [0-1].

		(The minimum and maximum of training and testing data is 398).
	"""
	return data / get_column_maxima()


def normalize_probabilities(predictions):
	"""
		Scale each ROW such that the SUM is exactly 1.
	"""
	return predictions / predictions.sum(axis = 1)[:,None]


def check_rows_normalized(predictions, epsilon = 1e-4):
	"""
		:return: True if rows add up to 1.
	"""
	return all(abs(predictions.sum(axis = 1) - 1) < epsilon)


def normalized_sum(v):
	"""
		:return: A copy of v that has a sum of 1.
	"""
	return array(v) / float64(array(v).sum())


if __name__ == '__main__':
	data = get_column_maxima() * uniform(size = (200, 93))
	ndata = normalize_data(data)
	print 'normalized data min/max:', ndata.min(), ndata.max()
	predictions = normalize_probabilities(uniform(size = (21, 9)))
	print 'probabilities normalized?', check_rows_normalized(predictions)


