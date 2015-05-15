
"""
	Prepare data for neural network use.
"""

from matplotlib.pyplot import subplots, show
from numpy import log10, zeros, where
from theano.scalar import float32
from settings import NCLASSES, VERBOSITY


def normalize_data(data, norms = None, use_log = True):
	if norms is None:
		norms = data.max(0).astype(float32)
	data = data / norms
	if use_log:
		data = log10(1 + 200 * data)
		if VERBOSITY >= 1:
			print 'normalizing positive data to [0, ...) with log scale'
	else:
		if VERBOSITY >= 1:
			print 'normalizing positive data to [0, 1] linearly'
	return data, norms


def conormalize_data(train, test, use_log = True):
	train, norms = normalize_data(train, use_log = use_log)
	if test is not None:
		test = normalize_data(test, use_log = use_log)[0]
	return train, test


def equalize_class_sizes(data, classes, min_size = 1929, class_count = NCLASSES):
	"""
		Equalize classes by removing samples to make them all the same size.

		:param min_size: The number of samples to use for each class.
		:return: trimmmed data and classes.
	"""
	if VERBOSITY >= 1:
		print 'balancing {0:d} classes by trimming all to {1:d} samples'.format(class_count, min_size)
	filter = zeros(classes.shape, dtype = bool)
	for cls in range(1, class_count + 1):
		this_cls = where(classes == cls)[0][:min_size]
		filter[this_cls] = True
	return data[filter], classes[filter]


if __name__ == '__main__':
	train_data, classes, features = normalize_data()
	print 'min  ', train_data.min()
	print 'max  ', train_data.max()
	print 'std  ', train_data.std()
	print 'std>0', train_data[train_data > 0].std()
	fig, ax = subplots()
	ax.hist(train_data.flat)
	show()


