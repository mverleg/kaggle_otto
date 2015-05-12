
"""
	Functions for generating extra features.
"""

from random import Random
from numpy import zeros, sort, where, cumsum, logical_and, concatenate, vstack
from settings import NFEATS, NCLASSES, SEED
from utils.loading import get_training_data
from utils.normalize import normalized_sum


class PositiveSparseFeatureGenerator(object):

	def __init__(self, data, labels, difficult_classes = (2, 3), extra_features = 57, multiplicity = 3,
			operation_probs = (0.3, 0.3, 0.2, 0.2), seed = 0):
		"""
			Feature generator to create positive features from positive, sparse data.

			:param difficult_classes: The labels of difficult classes to consider (labels, so starting from 1).
			:param extra_features: Number of features to create.
			:param multiplicity: The N new features are based on the N // multiplicity best old ones.
			:param operations: Probabilities for each operation:

			and   a + b
			xor   a + b if not (a and b) else 0
			-     max(a - b, 0)
			-     max(b - a, 0)
		"""
		assert extra_features // multiplicity >= 2, 'Need extra_features / multiplicity >= 2 or there will be not enough source features.'
		self.extra_features = extra_features
		self.operation_cumprobs = cumsum(normalized_sum(operation_probs))
		self.seed = SEED + seed
		self.sources = self.features_for_difficult_classes(data, labels,
			extra_feature_count = extra_features // multiplicity, difficult_classes = difficult_classes)

	def class_feature_count(self, train, labels):
		"""
			Find the number of times a feature is provided for each class (to find which features differentiate the difficult classes)

			Adapted from demo/class_feature_relation.py
		"""
		cnt = zeros((NFEATS, NCLASSES))
		for cls in range(0, NCLASSES):
			cnt[:, cls] = (train[cls + 1 == labels] != 0).sum(0)
		return cnt

	def features_for_difficult_classes(self, data, labels, extra_feature_count = 57, difficult_classes = (2, 3)):
		"""
			Select the features which are related to the difficult classes.

			:return: Indices of several features for difficult classes.
		"""
		all_cnts = self.class_feature_count(data, labels)
		difficult_cnts = 0
		for cls in difficult_classes:
			difficult_cnts += normalized_sum(all_cnts[:, cls - 1])
		cutoff = sort(difficult_cnts)[-extra_feature_count]
		return where(difficult_cnts >= cutoff)[0]

	def binary_feature(self, data1, data2, seed):
		operation = 0
		for operation, border in enumerate(self.operation_cumprobs):
			if seed < border:
				break
		if operation == 0:
			return data1 + data2
		elif operation == 1:
			feat = data1 + data2
			feat[logical_and(data1, data2)] = 0
			return feat
		elif operation == 2:
			""" It is important to use data1 <= data2 and not feat < 0, since uint8 overflows when < 0. """
			feat = data1 - data2
			feat[data1 <= data2] = 0
			return feat
		elif operation == 3:
			feat = data2 - data1
			feat[data2 <= data1] = 0
			return feat
		else:
			raise AssertionError('Operation with index {0:d} not found'.format(operation))

	def make_features(self, data):
		self.random = Random(self.seed)
		features = []
		for k in range(self.extra_features):
			(f1, f2), s = self.random.sample(self.sources, 2), self.random.random()
			feat = self.binary_feature(data[:, f1], data[:, f2], s)
			features.append(feat)
		return vstack(features).T

	def add_features(self, data):
		feats = self.make_features(data)
		return concatenate([data, feats], axis = 1)


if __name__ == '__main__':
	train_data, true_labels = get_training_data()[:2]
	gen = PositiveSparseFeatureGenerator(train_data, true_labels, difficult_classes = (2, 3), extra_features = 57, seed = 1)
	gen.add_features(train_data)
	gen.add_features(test_data)
	print 'old shape', train_data.shape
	print 'new shape', gen.add_features(train_data).shape


