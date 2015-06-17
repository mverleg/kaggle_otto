
from numpy import hstack, float64
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import LocallyLinearEmbedding, TSNE, MDS, SpectralEmbedding
from settings import RAW_NFEATS, VERBOSITY
from utils.loading import get_training_data
from utils.shuffling import shuffle


class ManifoldFeatures(BaseEstimator, TransformerMixin):

	def __init__(self, method_inst, only_upto = RAW_NFEATS, use_only = 5000):
		"""
			Abstract class for general manifold transforms.
		"""
		self.method_inst = method_inst
		self.use_only = use_only
		self.only_upto = only_upto

	def fit(self, X, y = None, **fit_params):
		if VERBOSITY >= 1:
			print '{0:s}: adding {1:d} features using {2:} samples'.format(self.__class__.__name__, self.method_inst.n_components, self.use_only or 'all')
		Xr = shuffle(X)[0][:self.use_only, :]
		self.method_inst.fit(Xr)
		return self

	def transform(self, X, y = None, copy = False):
		Xf = X[:, :self.only_upto]
		feats = self.method_inst.transform(Xf)
		return hstack((X, feats,))


class LLEFeatures(ManifoldFeatures):
	"""
		Locally Linear Embedding
	"""
	def __init__(self, extra_featurs, n_neighbors = 50, eigen_solver = 'auto', method = 'standard', only_upto = RAW_NFEATS, use_only = 5000):
		inst = LocallyLinearEmbedding(n_neighbors = n_neighbors, n_components = extra_featurs, eigen_solver = eigen_solver, method = method)
		super(LLEFeatures, self).__init__(method_inst = inst, only_upto = only_upto, use_only = use_only)


class TSNEFeatures(ManifoldFeatures):
	"""
		t-Distributed Stochastic Neighbor Embedding
	"""
	def __init__(self, extra_featurs, perplexity = 50., only_upto = RAW_NFEATS, use_only = 5000):
		inst = TSNE(n_components = extra_featurs, perplexity = perplexity)
		super(TSNEFeatures, self).__init__(method_inst = inst, only_upto = only_upto, use_only = use_only)


class MDFeatures(ManifoldFeatures):
	"""
		Multidimensional scaling
	"""
	def __init__(self, extra_featurs, metric = True, n_init = 4, max_iter = 300, eps = 0.001, dissimilarity = 'euclidean', only_upto = RAW_NFEATS, use_only = 5000):
		inst = MDS(n_components = extra_featurs, metric=True, n_init = n_init, max_iter = max_iter, eps = eps, dissimilarity = dissimilarity)
		super(MDFeatures, self).__init__(method_inst = inst, only_upto = only_upto, use_only = use_only)


class SEFeatures(ManifoldFeatures):
	"""
		SpectralEmbedding(n_components=17,n_neighbors=50)
	"""
	def __init__(self, extra_featurs, n_neighbors = 50, only_upto = RAW_NFEATS, use_only = 5000):
		inst = SpectralEmbedding(n_components = extra_featurs, n_neighbors = n_neighbors)
		super(SEFeatures, self).__init__(method_inst = inst, only_upto = only_upto, use_only = use_only)


if __name__ == '__main__':
	data = get_training_data()[0].astype(float64)
	trans = ManifoldFeatures(LocallyLinearEmbedding(n_neighbors = 50, n_components = 10, eigen_solver = 'auto', method = 'standard'))
	print trans.fit_transform(data, use_only = 1000).shape


