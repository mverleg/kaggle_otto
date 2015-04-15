
"""
	Load train and test data.

	Performance for me (Mark):
	- first load with this method:   9.5s
	- later loads with this method:  0.3s
	- any load with previous method: 8.8s
	So the initial loading can be faster, but there's no need.
"""

from tempfile import gettempdir
from numpy import array, uint16, load, save, empty
from collections import OrderedDict
from os.path import join
from settings import VERBOSITY, TRAIN_DATA_PATH, TEST_DATA_PATH


TRAINSIZE = 61878
TESTSIZE = 144368
NFEATS = 93


def load_training_data(filepath = TRAIN_DATA_PATH):
	"""
		Load training data.

		:return: data[ndarray], samples:classes[OrderedDict], features[list]

		Notes:
		* I tried using Pandas to have a dataframe with labels, but it seems I can't skip class label column. -Mark
		* The samples are numbered from 1 because the data file does that. Sorry.
	"""
	with open(filepath, 'r') as fh:
		cells = [line.split(',') for line in fh.read().splitlines()]
	features = cells[0][1:-1]
	classes = OrderedDict()
	data = empty((TRAINSIZE, NFEATS), dtype = uint16)
	for k, row in enumerate(cells[1:]):
		classes[int(row[0])] = uint16(row[-1].split('_')[-1])
		data[k, :] = row[1:-1]
	return data, classes, features


def load_testing_data(filepath = TEST_DATA_PATH):
	"""
		Load testing data.

		:return: data[ndarray], features[list]
	"""
	with open(filepath, 'r') as fh:
		cells = [line.split(',') for line in fh.read().splitlines()]
	features = cells[0][1:]
	data = empty((TESTSIZE, NFEATS), dtype = uint16)
	for k, row in enumerate(cells[1:]):
		data[k, :] = row[1:]
	return data, features


def get_training_data(filepath = TRAIN_DATA_PATH):
	"""
		Gets the training data from the CSV file, caching it in temporary directory for speed.
	"""
	try:
		data = load(join(gettempdir(), 'cache_train_data.npy'))
		features = load(join(gettempdir(), 'cache_train_features.npy'))
		classvals = load(join(gettempdir(), 'cache_train_classes.npy'))
		classes = OrderedDict((k + 1, val) for k, val in enumerate(classvals))
		if VERBOSITY >= 1:
			print 'loaded train data from cache in "{0:s}"'.format(gettempdir())
	except:
		data, classes, features = load_training_data(filepath = filepath)
		save(join(gettempdir(), 'cache_train_data.npy'), data)
		save(join(gettempdir(), 'cache_train_features.npy'), features)
		save(join(gettempdir(), 'cache_train_classes.npy'), array(classes.values()))
		if VERBOSITY >= 1:
			print 'loaded train data directly'
	return data, classes, features


def get_testing_data(filepath = TEST_DATA_PATH):
	"""
		Gets the test data from the CSV file, caching it in temporary directory for speed.
	"""
	try:
		data = load(join(gettempdir(), 'cache_test_data.npy'))
		features = load(join(gettempdir(), 'cache_test_features.npy'))
		if VERBOSITY >= 1:
			print 'loaded test data from cache in "{0:s}"'.format(gettempdir())
	except:
		data, features = load_testing_data(filepath = filepath)
		save(join(gettempdir(), 'cache_test_data.npy'), data)
		save(join(gettempdir(), 'cache_test_features.npy'), features)
		if VERBOSITY >= 1:
			print 'loaded test data directly'
	return data, features


"""
	Sorry for removing the TEST, TRAIN and LABELS globals. It is a bad idea to load train and test data if only one of them might be needed.
"""


if __name__ == '__main__':
	train_data, classes, features = get_training_data()
	test_data, features = get_testing_data()


