
"""
	General code to optimize parameters to a model.
"""

from collections import Iterable, OrderedDict
from json import dumps, loads
from os import makedirs
from os.path import isdir, join
from numpy import zeros, prod, float64
from grid import to_coordinate
from settings import OPTIMIZE_RESULTS_DIR, VERBOSITY
from validation.crossvalidate import Validator


def is_nonstr_iterable(obj):
	"""
		http://stackoverflow.com/questions/19943654/type-checking-an-iterable-type-that-is-not-a-string
	"""
	return not isinstance(obj, str) and isinstance(obj, Iterable)


class GridOptimizer(object):

	def __init__(self, validator, use_caching = True, prefix = None, **params):
		"""
			Create a grid optimizer to do a grid search for optimal parameter vaues.

			:param validator: A validator to use for getting the score for each parameter.
			:param params: The parameters you want to compare, with a list of the values to try:

				GridOptimizer(validator, learning_rate = [1, 0.1, 0.01], hidden_layer_size = [30, 50], momentum = 0.9)

			:param prefix: Prefix for cache files (can be left empty).
			Results are cached in the directory set by settings.OPTIMIZE_RESULTS_DIR
		"""
		assert isinstance(validator, Validator), 'Argument "validator" should be an instantiated Validator (not "{0:s}").'.format(type(validator))
		self.validator = validator
		self.rounds = self.validator.rounds
		self.use_caching = bool(use_caching)
		self.prefix = prefix or ''
		self.fixed_params = {key: val for key, val in params.items() if not is_nonstr_iterable(val)}
		self.iter_params = OrderedDict((key, val) for key, val in params.items() if is_nonstr_iterable(val))
		self.labels, self.indices = zip(*[(key, val) for key, val in params.items() if is_nonstr_iterable(val)])
		self.dims = tuple(len(li) for li in self.indices)
		self.results = zeros(self.dims + (self.rounds, 3,), dtype = float64)
		self.results_added = 0
		print 'grid optimize: {0:s} comparisons x {1:d} rounds = {2:d} iterations'.format(' x '.join(str(d) for d in self.dims), self.rounds, prod(self.dims) * self.rounds)
		try:
			makedirs(OPTIMIZE_RESULTS_DIR)
		except OSError:
			""" Probably already exists; ignore it. """

	def params_name(self, params):
		params = OrderedDict(sorted(params.items()))
		return (
			self.prefix + '_'.join('{0:s}-{1:}'.format(key, val) for key, val in params.items()),
			', '.join('{0:s} = {1:}'.format(key, val) for key, val in params.items()),
		)

	def store_results(self, filepath, logloss, accuracy, duration):
		with open(filepath, 'w+') as fh:
			fh.write(dumps({'logloss': logloss, 'accuracy': accuracy, 'duration': duration}, indent = 4, sort_keys = True))

	def load_results(self, filepath):
		"""
			:return: (logloss, accuracy, duration) tuple
		"""
		with open(filepath, 'r') as fh:
			d = loads(fh.read())
		return d['logloss'], d['accuracy'], d['duration'],

	def get_single_batch(self, params, round, name):
		"""
			Return a single batch (intended for internal use).
		"""
		train_data, train_classes, test_data = self.validator.start_round(round)
		if VERBOSITY >= 2:
			print 'calculate: %s, round #%d/%d' % (name, round + 1, self.rounds)
		return params, train_data, train_classes, test_data

	def yield_batches(self):
		"""
			Iterator that goes over the different training parameters and the different cross validation rounds for each of them.

			:return: An iterator with (parameters, train_data, train_classes, test_data) tuple on each iteration.
		"""
		for p in range(prod(self.dims)):
			""" Every combination of parameters. """
			coord = to_coordinate(p, self.dims)
			params = {self.labels[d]: self.indices[d][k] for d, k in enumerate(coord)}
			params.update(self.fixed_params)
			self.validator.reset()
			filename, dispname = self.params_name(params)
			print 'calculating {0:d} rounds for parameters {1:s}'.format(self.rounds, dispname)
			for round in range(self.rounds):
				if self.use_caching:
					try:
						""" Try to load cache. """
						results = self.load_results(join(OPTIMIZE_RESULTS_DIR, filename))
					except IOError:
						""" No cache; yield the data (storage happens elsewhere). """
						yield self.get_single_batch(params, round, dispname)
					else:
						""" Cache loaded; handle. """
						self.add_results(*results)
						if VERBOSITY >= 2:
							print 'cache: %s, round #%d/%d' % (dispname, round + 1, self.rounds)
				else:
					yield self.get_single_batch(params, round, dispname)

	def add_results(self, logloss, accuracy, duration):
		param_index = self.results_added // self.rounds
		round_index = self.results_added % self.rounds
		coord = to_coordinate(param_index, self.dims)
		arr = self.results
		for k in coord:
			arr = arr[k]
		arr[round_index][:] = logloss, accuracy, duration
		self.results_added += 1

	def register_results(self, prediction):
		"""
			Register results of an optimization round.

			:param prediction: SxC array with predicted probabilities, with each row corresponding to a test data sample and each column corresponding to a class.
		"""
		assert self.results_added < prod(self.dims) * self.rounds, 'There are already {0:d} results for {1:d} slots.'.format(self.results_added + 1, prod(self.dims) * self.rounds)
		coord = to_coordinate(self.results_added // self.rounds, self.dims)
		params = {self.labels[d]: self.indices[d][k] for d, k in enumerate(coord)}
		filename, dispname = self.params_name(params)
		results = self.validator.add_prediction(prediction)
		self.add_results(*results)
		self.store_results(join(OPTIMIZE_RESULTS_DIR, filename), *results)
		return results

	def print_plot_results(self):
		"""
			Once all results are calculated, print statistics and plot graphs to see the performance.
		"""
		#coord = to_coordinate(p, self.dims)
		#params = {self.labels[d]: self.indices[d][k] for d, k in enumerate(coord)}
		#params.update(self.fixed_params)
		if len(self.dims) == 0:
			print 'There are no parameters that have different values; nothing to compare.'
		elif len(self.dims) == 1:
			print 'Showing results for "{0:s}"'.format(self.labels[0])
		elif len(self.dims) == 2:
			print 'Showing results for "{0:s}" and "{1:s}"'.format(self.labels[0], self.labels[1])
		else:
			print 'There are more than two parameters to compare; no visualization options.'
		print 'The minimum ...'


