
"""
	http://scikit-learn.org/stable/modules/outlier_detection.html

	http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html
"""

from hashlib import sha1
from numpy import ones
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from settings import NCLASSES, SEED
from tempfile import gettempdir
from numpy import load, save
from os.path import join
from settings import VERBOSITY, TRAIN_DATA_PATH
from utils.loading import get_training_data


def get_filtered_data(cut_outlier_frac = 0.05, method = 'EE', filepath = TRAIN_DATA_PATH):
	"""
		Load and filter data and classes to remove an approximate fraction of outliers.

		:param cut_outlier_frac: fraction of outliers to remove (approximate, especially for OCSVM)
		:param method: either 'EE' for Elliptical Envelope or 'OCSVM' for One Class Support Vector Machines
		:return: filtered train data, train classes and features
	"""
	data, classes, features = get_training_data()
	hash = '{0:s}_{1:04d}_all.npy'.format(method, int(cut_outlier_frac * 1000))
	data, classes = filter_data_cache(data, classes, hash = hash, method = method, cut_outlier_frac = cut_outlier_frac)
	return data, classes, features


def filter_data(data, classes, cut_outlier_frac = 0.05, method = 'EE'):
	"""
		Filteres ndarrays data and classes to remove an approximate fraction of outliers.

		:param data: ndarray with data (2D) where samples are rows
		:param classes: ndarray with classes (1D)
		:param cut_outlier_frac: fraction of outliers to remove (approximate, especially for OCSVM)
		:param method: either 'EE' for Elliptical Envelope or 'OCSVM' for One Class Support Vector Machines
		:return: filtered data and classes
	"""
	hash = '{0:s}_{1:04d}_{2:.10s}.npy'.format(method, int(cut_outlier_frac * 1000), sha1((data.tostring())).hexdigest())
	return filter_data_cache(data, classes, hash = hash, method = method, cut_outlier_frac = cut_outlier_frac)


def make_filtered_data(train_data, true_classes, cut_outlier_frac, detector):
	""" Not intended for direct use. """
	print 'removing outliers'
	assert 0 <= cut_outlier_frac <= 1. + 1e-6
	keep = ones(true_classes.shape, dtype = bool)
	for cls in range(1, NCLASSES + 1):
		if VERBOSITY:
			print 'removing outliers from class {0:d}'.format(cls)
		cls_train = train_data[true_classes == cls, :]
		transformed = PCA(n_components = 40, copy = True, whiten = False).fit(cls_train).transform(cls_train)
		decisions = detector.fit(transformed).decision_function(transformed)
		keep[true_classes == cls] *= (decisions > 0)
	return train_data[keep, :], true_classes[keep]


def filter_data_cache(data, classes, hash, method, cut_outlier_frac):
	""" Not intended for direct use. """
	print join(gettempdir(), 'cache_data_nooutliers_{0:s}.npy'.format(hash))
	try:
		data = load(join(gettempdir(), 'cache_data_nooutliers_{0:s}.npy'.format(hash)))
		classes = load(join(gettempdir(), 'cache_classes_nooutliers_{0:s}.npy'.format(hash)))
		if VERBOSITY >= 1:
			print 'loaded filtered train data from cache in "{0:s}" with outlying {1:.1f}% of data removed.'.format(gettempdir(), 100 * cut_outlier_frac)
	except IOError:
		if method == 'EE':
			detector = EllipticEnvelope(contamination = cut_outlier_frac, random_state = SEED)
		elif method == 'OCSVM':
			detector = OneClassSVM(nu = 0.95 * cut_outlier_frac + 0.05, kernel = 'rbf', gamma = 0.1)
		else:
			raise AssertionError('Only methods "EE" (Eliptical Envelope) and "OCSVM" (One Class Support Vector Machine) are available at this time.')
		data, classes = make_filtered_data(data, classes, cut_outlier_frac = cut_outlier_frac, detector = detector)
		save(join(gettempdir(), 'cache_data_nooutliers_{0:s}.npy'.format(hash)), data)
		save(join(gettempdir(), 'cache_classes_nooutliers_{0:s}.npy'.format(hash)), classes)
		if VERBOSITY >= 1:
			print 'loaded filtered train data directly with outlying {0:.1f}% of data removed.'.format(100 * cut_outlier_frac)
	return data, classes


