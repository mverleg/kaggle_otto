
"""
	Functions for generating extra features.
"""

from random import Random
from matplotlib.pyplot import subplots, show
from numpy import zeros, sort, where, cumsum, logical_and, concatenate, vstack, isnan, any, sqrt, array, hstack
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from settings import NCLASSES, SEED, VERBOSITY, RAW_NFEATS
from utils.loading import get_training_data
from utils.normalize import normalized_sum


DIFFICULT_CLASSES_MIX = {
	(2, 3): 0.3,
	(2, 3, 4): 0.3,
	(1, 9): 0.4,
}


def chain_feature_generators(train_data, true_labels, test_data, classes = DIFFICULT_CLASSES_MIX, extra_features = 57,
		multiplicity = None, operation_probs = (0.3, 0.3, 0.2, 0.2), seed = 0):
	"""
		Apply several feature generators for different pairs of difficult classes.

		:param train_data: The data on which the feature generator bases the choice of features. Also immediately creates features for this data.
		:param test_data: This data is only used to generate features.
		:param classes: A dictionary with contributions from different pairs of classes.
		:return: The augmented train and test data.
	"""
	if not extra_features:
		return train_data, test_data
	gen = PositiveSparseRowFeatureGenerator(extra_features = min(extra_features, 16))
	train_data = gen.fit_transform(train_data)
	test_data = gen.fit_transform(test_data)
	extra_features -= min(extra_features, 16)
	if multiplicity is None:
		multiplicity = max(min(extra_features // 10, 3), 1)
	assert abs(sum(classes.values()) - 1) < 1e-6,  'Class contributions should be normalized.'
	if VERBOSITY >= 1:
		print 'creating {0:d} extra features for {1:d} groups of classes'.format(extra_features, len(classes))
	class_counts = {key: int(extra_features * val) for key, val in classes.items()}
	for key in class_counts.keys():
		if sum(class_counts.values()) >= extra_features:
			break
		class_counts[key] += 1
	for offset, (difficult, contribution) in enumerate(classes.items()):
		gen = PositiveSparseFeatureGenerator(difficult_classes = difficult,
			extra_features = int(round(extra_features * contribution)), multiplicity = multiplicity,
			seed = offset + 100 * seed)
		gen.fit(train_data, true_labels)
		if test_data is not None:
			train_data, test_data = gen.add_features_tt(train_data, test_data)
		else:
			train_data = gen.add_features(train_data)
	return train_data, test_data


class PositiveSparseFeatureGenerator(BaseEstimator, TransformerMixin):

	def __init__(self, difficult_classes = (2, 3), extra_features = 57, multiplicity = 3,
			operation_probs = (0.3, 0.3, 0.4), only_upto = RAW_NFEATS, seed = 0):
		"""
			Feature generator to create positive features from positive, sparse data.

			:param difficult_classes: The labels of difficult classes to consider (labels, so starting from 1).
			:param extra_features: Number of features to create.
			:param multiplicity: The N new features are based on the N // multiplicity best old ones.
			:param operations: Probabilities for each operation:

			and   (a + b + c + ...) / N
			xor   a or b iff one of them is set (anything after b ignored)
			+-    max(a - b + c - d + ..., 0)
		"""
		assert extra_features // multiplicity >= 2, 'Need extra_features / multiplicity >= 2 or there will be not enough source features (for {0:d} / {1:d}).'.format(extra_features, multiplicity)
		self.extra_features = extra_features
		self.multiplicity = multiplicity
		self.source_count = self.extra_features // self.multiplicity
		self.difficult_classes = difficult_classes
		self.only_upto = only_upto
		self.operation_probs = operation_probs
		self.operation_cumprobs = cumsum(normalized_sum(operation_probs))
		self.seed = SEED + seed

	def fit(self, X, y, **fit_params):
		self.sources = self.features_for_difficult_classes(X[:, :self.only_upto], y,
			difficult_feature_count = self.source_count, difficult_classes = self.difficult_classes)
		return self

	def transform(self, X, y = None, copy = False):
		if VERBOSITY >= 1:
			print 'adding {0:d} features for classes {1:s}'.format(self.extra_features, self.difficult_classes)
		return self.add_features(X)

	def get_params(self, deep = True):
		return {
			'difficult_classes': self.difficult_classes,
			'extra_features': self.extra_features,
			'multiplicity': self.multiplicity,
			'operation_probs': self.operation_probs,
			'only_upto': self.only_upto,
			'seed': self.seed - SEED,
		}

	def class_feature_count(self, train, labels):
		"""
			Find the number of times a feature is provided for each class (to find which features differentiate the difficult classes)

			Adapted from demo/class_feature_relation.py
		"""
		cnt = zeros((train.shape[1], NCLASSES))
		for cls in range(0, NCLASSES):
			cnt[:, cls] = (train[cls + 1 == labels] != 0).sum(0)
		return cnt

	def features_for_difficult_classes(self, data, labels, difficult_feature_count=57, difficult_classes=(2, 3)):
		"""
			Select the features which are related to the difficult classes.

			:return: Indices of several features for difficult classes.
		"""
		all_cnts = self.class_feature_count(data, labels)
		difficult_cnts = 0
		for cls in difficult_classes:
			difficult_cnts += normalized_sum(all_cnts[:, cls - 1])
		cutoff = sort(difficult_cnts)[-difficult_feature_count]
		return where(difficult_cnts >= cutoff)[0]

	def get_operation(self, seed):
		for operation, border in enumerate(self.operation_cumprobs):
			if seed < border:
				break
		else:
			raise AssertionError('Operation not found for seed {0}'.format(seed))
		return operation

	def binary_feature(self, data1, data2, seed):
		operation = self.get_operation(seed)
		if operation == 0:
			return (data1 + data2 + 1) // 2
		elif operation == 1:
			feat = data1 + data2
			feat[logical_and(data1, data2)] = 0
			return feat
		elif operation == 2:
			""" It is important to use data1 <= data2 and not feat < 0, since uint8 overflows when < 0. """
			feat = data1 - data2
			feat[data1 <= data2] = 0
			return feat
		else:
			raise AssertionError('Binary operation with index {0:d} not found'.format(operation))

	def poly_feature(self, datas, seed):
		operation = self.get_operation(seed)
		if operation == 0:
			feat = sum(datas) // len(datas)
			return feat
		elif operation == 1:
			feat = datas[0] + datas[1]
			feat[logical_and(datas[0], datas[1])] = 0
			return feat
		elif operation == 2:
			""" It is important to use data1 <= data2 and not feat < 0, since uint8 overflows when < 0. """
			pos, neg = sum(datas[0::2]), sum(datas[1::2])
			feat = (pos - neg) // int(sqrt(len(datas)))
			feat[pos <= neg] = 0
			return feat
		else:
			raise AssertionError('Poly operation with index {0:d} not found'.format(operation))

	def make_features(self, data):
		# call .fit() first
		self.random = Random(self.seed)
		features = []
		for k in range(self.extra_features):
			cnt = int(self.random.randint(2, len(self.sources))**0.6 + 1)
			indices = self.random.sample(self.sources, cnt)
			seed = self.random.random()
			feat = self.poly_feature([data[:, index] for index in indices], seed)
			features.append(feat)
			assert feat.max() > 0, 'Extra feature #{0:d} seems to be all zeros among {3:d} samples, which should not happen (it also causes problems with some classifiers). Feature operation {1:d} based on {2:s}'.format(k, self.get_operation(seed), str(indices), data.shape[0])
		return vstack(features).T

	def add_features(self, data):
		# call .fit() first
		feats = self.make_features(data)
		return concatenate([data, feats], axis = 1)

	def add_features_tt(self, train, labels, test):
		# call .fit() first
		return self.transform(train), self.transform(test)


class PositiveSparseRowFeatureGenerator(BaseEstimator, TransformerMixin):

	def __init__(self, extra_featurs = None, only_upto = RAW_NFEATS):
		"""
			Add some row-based extra features.

			:param extra_featurs: How many extra features to add at most (there is a limited amount).
			:param only_upto:

			Note that this doesn't use any learning, so it can be applied before training/validation to all data.
		"""
		self.extra_featurs = extra_featurs
		self.only_upto = only_upto

	def fit(self, X, y = None, **fit_params):
		return self

	#todo: should y = None be here?
	def transform(self, X, y = None, copy = False):
		if VERBOSITY >= 1:
			if self.extra_featurs is None:
				print 'adding all extra row features'.format(self.extra_featurs)
			else:
				print 'adding upto {0:d} row features'.format(self.extra_featurs)
		Xf = X[:, :self.only_upto]
		max_poss = Xf.argpartition(-5, axis = 1)[:, -5:]
		max_poss = array(list(max_poss[k, Xf[k, max_poss[k, :]].argsort()[::-1]] for k in range(Xf.shape[0])))
		feats = array([
			Xf.sum(1),
			Xf.max(1),
			Xf.argmax(1),
			max_poss[:, 0],
			max_poss[:, 1],
			max_poss[:, 2],
			max_poss[:, 3],
			max_poss[:, 4],
			(Xf == 0).sum(1),
			(Xf == 1).sum(1),
			(Xf == 2).sum(1),
			(Xf == 3).sum(1),
			((Xf >  3) * (Xf <=  7)).sum(1),
			((Xf >  7) * (Xf <= 15)).sum(1),
			((Xf > 15) * (Xf <= 30)).sum(1),
			((Xf > 30) * (Xf <= 70)).sum(1),
			(Xf > 70).sum(1),
			#todo: maybe manifold algorithms, but slow and non-integer
		]).T
		if self.extra_featurs is not None and feats.shape[1] < self.extra_featurs:
			print 'WARNING: {0:s} added only {1:d} features instead of {2:d} requested'.format(self.__class__.__name__, feats.shape[1], self.extra_featurs)
		if self.extra_featurs is None:
			return hstack((X, feats))
		return hstack((X, feats[:, :self.extra_featurs]))


class DistanceFeatureGenerator(BaseEstimator, TransformerMixin):
	"""
		Create extra features that are the distances to the nearest neighbors from each cluster class.
	"""
	def __init__(self, n_neighbors = 4, distance_p = 1):
		self.knn = [None] * NCLASSES
		for cls in range(1, NCLASSES + 1):
			self.knn[cls - 1] = KNeighborsClassifier(
				n_neighbors = n_neighbors,
				p = distance_p,
			)

	def fit(self, X, y):
		for cls in range(1, NCLASSES + 1):
			f = (y == cls)
			self.knn[cls - 1].fit(X[f], zeros((f.sum(),)))
		return self

	def transform(self, X):
		if VERBOSITY >= 1:
			print 'creating class distance features for {0:d}x{1:d} data'.format(*X.shape)
		feats = []
		for cls in range(1, NCLASSES + 1):
			if VERBOSITY >= 2:
				print ' creating class distance features for class {0}'.format(cls)
			dist, indx1 = self.knn[cls - 1].kneighbors(X, return_distance = True)
			feats.append(dist.sum(1))
		return hstack((X, array(feats).T))


if __name__ == '__main__':
	train_data, true_labels = get_training_data()[:2]
	augmented_data, duplicate_data = chain_feature_generators(train_data, true_labels, train_data, extra_features = 163, multiplicity = 3, seed = 1)
	print 'old shape', train_data.shape
	print 'new shape', augmented_data.shape
	fig, (ax1, ax2) = subplots(2)
	im = ax1.imshow(augmented_data[:100, :])
	fig.colorbar(im, ax = ax1)
	ax2.bar(range(augmented_data.shape[1]), augmented_data.mean(0))
	show()

