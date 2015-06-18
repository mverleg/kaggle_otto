
"""
	Functions for generating distance related sklearn stuff.
"""

from numpy import zeros, array, hstack
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from settings import NCLASSES, VERBOSITY, RAW_NFEATS


class DistanceBase(BaseEstimator):
	"""
		Create extra features that are the distances to the nearest neighbors from each cluster class (so NOT the number of neighbours!).
	"""
	def __init__(self, n_neighbors = 4, distance_p = 1, only_upto = RAW_NFEATS, nr_classes = NCLASSES):
		self.set_params({
			'n_neighbors': n_neighbors,
			'distance_p':  distance_p,
			'only_upto': only_upto,
			'nr_classes': nr_classes,
		})

	def fit(self, X, y):
		self.knn = [None] * self.nr_classes
		for cls in range(1, self.nr_classes + 1):
			self.knn[cls - 1] = KNeighborsClassifier(
				n_neighbors = self.n_neighbors,
				p = self.distance_p,
			)
		for cls in range(1, self.nr_classes + 1):
			f = (y == cls)
			self.knn[cls - 1].fit(X[f, :self.only_upto], zeros((f.sum(),)))
		return self

	def get_distances(self, X):
		dists = []
		for cls in range(1, self.nr_classes + 1):
			if VERBOSITY >= 2:
				print ' creating class distance features for class {0}'.format(cls)
			dist, indx1 = self.knn[cls - 1].kneighbors(X[:, :self.only_upto], return_distance = True)
			dists.append(dist.sum(1))
		return array(dists).T

	def get_params(self, deep = None):
		return {
			'n_neighbors': self.n_neighbors,
			'distance_p':  self.distance_p,
			'only_upto': self.only_upto,
			'nr_classes': self.nr_classes,
		}

	def set_params(self, params):
		self.n_neighbors = params['n_neighbors']
		self.distance_p = params['distance_p']
		self.only_upto = params['only_upto']
		self.nr_classes = params['nr_classes']
		return self


class DistanceFeatureGenerator(DistanceBase, TransformerMixin):
	"""
		Create extra features that are the distances to the nearest neighbors from each cluster class (so NOT the number of neighbours!).
	"""

	def transform(self, X):
		if VERBOSITY >= 1:
			print 'creating class distance features for {0:d}x{1:d} data'.format(*X.shape)
		feats = self.get_distances(X)
		return hstack((X, array(feats).T))



class DistanceClassifier(DistanceBase, ClassifierMixin):
	"""
		Classifier based on average distance of neighbours per class (so uses the distance, unlike kNN which uses number of neighbours).
	"""

	def predict_proba(self, X):
		dists = feats = self.get_distances(X)
		return dists / dists.sum(1, keepdims = True)

	def predict(self, X):
		return self.predict_proba(X).argmax(1)



"""
class DistanceFeatureGenerator(KNeighborsClassifier):
	def __init__(self, only_upto = RAW_NFEATS, *args, **kwargs):
		super(DistanceFeatureGenerator, self).__init__(self, *args, **kwargs)
		self.only_upto = only_upto

	def fit(self, X, y):
		super(DistanceFeatureGenerator, self).fit(X[:, :self.only_upto], y)

	def transform(self, X):
		super(DistanceFeatureGenerator, self).transform(X[:, :self.only_upto])
"""