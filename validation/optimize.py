
"""
	General code to optimize parameters to a model.
"""

from hashlib import sha1
from collections import Iterable, OrderedDict
from json import dumps, loads
from os.path import join
from sys import stdout
from matplotlib.pyplot import show
from numpy import zeros, prod, float64, unravel_index, ravel_multi_index, where, size
from settings import OPTIMIZE_RESULTS_DIR, VERBOSITY, AUTO_IMAGES_DIR
from validation.crossvalidate import Validator
from validation.views import compare_bars, compare_plot, compare_surface


class GridOptimizer(object):

	def __init__(self, validator, use_caching = True, prefix = None, **params):
		"""
			Create a grid optimizer to do a grid search for optimal parameter vaues.

			:param validator: A validator to use for getting the score for each parameter.
			:param use_cahcing: Store the results to make comparison faster next time.
			:param prefix: Prefix for cache files (can be left empty).
			:param params: The parameters you want to compare, with a list of the values to try:

				GridOptimizer(validator, learning_rate = [1, 0.1, 0.01], hidden_layer_size = [30, 50], momentum = 0.9)

			Results are cached in the directory set by settings.OPTIMIZE_RESULTS_DIR.

			The code was not designed for iterable parameters. You can try to put them in another iterable, but a simple mapping may be easier (e.g. m = {1: [...], 2: [...]} , pass [1, 2] to GridOptimizer and let the code use m[param]).
		"""
		#print 'Note: there is now a parallel version of GridOptimizer. It works a little differently, but can be much faster.'
		assert isinstance(validator, Validator), 'Argument "validator" should be an instantiated Validator (not "{0:s}").'.format(type(validator))
		self.validator = validator
		self.rounds = self.validator.rounds
		self.use_caching = bool(use_caching)
		self.prefix = prefix or ''
		self.fixed_params = {key: val for key, val in params.items() if not is_nonstr_iterable(val)}
		iter_params = OrderedDict((key, sorted(val)) for key, val in params.items() if is_nonstr_iterable(val))
		try:
			self.labels, self.values = zip(*[(key, val) for key, val in iter_params.items() if is_nonstr_iterable(val)])
		except ValueError:
			self.labels, self.values = tuple(), tuple()
		self.dims = tuple(len(li) for li in self.values)
		self.results = zeros(self.dims + (self.rounds, 3,), dtype = float64)
		self.results_added = 0
		print 'grid optimize: {0:s} comparisons x {1:d} rounds = {2:d} iterations'.format(' x '.join(unicode(d) for d in self.dims), self.rounds, prod(self.dims, dtype = int) * self.rounds)

	def get_single_batch(self, params, round, name):
		"""
			Return a single batch (intended for internal use).
		"""
		train_data, train_classes, test_data = self.validator.start_round(round)
		if VERBOSITY >= 2:
			print 'calculate: %s, round #%d/%d' % (name, round + 1, self.rounds)
		return params, train_data, train_classes, test_data

	def yield_batches(self, print_current_parameters = True):
		"""
			Iterator that goes over the different training parameters and the different cross validation rounds for each of them.

			:return: An iterator with (parameters, train_data, train_classes, test_data) tuple on each iteration.
		"""
		for p in range(prod(self.dims, dtype = int)):
			""" Every combination of parameters. """
			coord = unravel_index(p, self.dims) if self.dims else tuple()
			params = {self.labels[d]: self.values[d][k] for d, k in enumerate(coord)}
			params.update(self.fixed_params)
			self.validator.reset()
			filename, dispname = params_name(params, self.prefix)
			if print_current_parameters:
				print 'calculating {0:d} rounds for parameters {1:s}'.format(self.rounds, dispname)
			for round in range(self.rounds):
				if self.use_caching:
					try:
						""" Try to load cache. """
						res = load_results(join(OPTIMIZE_RESULTS_DIR, filename + 'r{0:d}.result'.format(round)))
					except IOError as err:
						""" No cache; yield the data (storage happens elsewhere). """
						yield self.get_single_batch(params, round, dispname)
					else:
						""" Cache loaded; handle. """
						self.add_results(self.results_added, *res)
						if print_current_parameters:
							print 'cache: %s, round #%d/%d' % (dispname, round + 1, self.rounds)
				else:
					yield self.get_single_batch(params, round, dispname)

	def add_results(self, index, logloss, accuracy, duration):
		param_index = index // self.rounds
		round_index = index % self.rounds
		coord = unravel_index(param_index, self.dims) if self.dims else tuple()
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
		coord = unravel_index(self.results_added // self.rounds, self.dims) if self.dims else tuple()
		round = self.results_added % self.rounds
		params = {self.labels[d]: self.values[d][k] for d, k in enumerate(coord)}
		params.update(self.fixed_params)
		filename, dispname = params_name(params, self.prefix)
		res = self.validator.add_prediction(prediction)
		self.add_results(self.results_added, *res)
		store_results(join(OPTIMIZE_RESULTS_DIR, filename + 'r{0:d}.result'.format(round)), *res)
		return res

	def print_top(self, topprint):
		"""
			Print the lowest logloss results.
		"""
		if not self.dims:
			return
		logloss_slice = [slice(None)] * len(self.dims) + [slice(None), 0]
		logloss_all = self.results[logloss_slice]
		logloss_mean = logloss_all.mean(len(self.dims))
		logloss_cutoff = sorted(logloss_mean.flat, reverse = False)[min(topprint, size(logloss_mean) - 1)]
		min_coords = zip(*where(logloss_mean <= logloss_cutoff))
		min_coords = sorted(min_coords, key = lambda pos: logloss_mean.flat[ravel_multi_index(pos, self.dims)])
		stdout.write('pos     loss      {0:s}\n'.format('  '.join('{0:16s}'.format(label.replace('_', ' '))[-16:] for label in self.labels)))
		for pos, min_coord in enumerate(min_coords):
			stdout.write('{0:2d}  {1:8.4f}     '.format(pos + 1,
				logloss_mean.flat[ravel_multi_index(min_coord if self.dims else tuple(), self.dims)],
			))
			for k, j in enumerate(min_coord):
				stdout.write('  {0:16s}'.format(unicode(self.values[k][j])))
			stdout.write('\n')

	def print_plot_results(self, topprint = 12, save_fig_basename = None):
		"""
			Once all results are calculated, print statistics and plot graphs to see the performance.
		"""
		if len(self.dims) == 0:
			print 'There are no parameters that have different values; nothing to compare.'
		elif len(self.dims) == 1:
			print 'Showing results for "{0:s}"'.format(self.labels[0])
			if all(is_number(param) for param in sum(self.values, [])):
				fig, axi = compare_plot(self.results, self.labels, self.values)
				if save_fig_basename:
					fig.savefig(join(AUTO_IMAGES_DIR, '{0:s}_plot.png'.format(save_fig_basename)))
			fig, axi = compare_bars(self.results, self.labels, self.values)
			if save_fig_basename:
				fig.savefig(join(AUTO_IMAGES_DIR, '{0:s}_bars.png'.format(save_fig_basename)))
		elif len(self.dims) == 2:
			print 'Showing results for "{0:s}" and "{1:s}"'.format(self.labels[0], self.labels[1])
			fig, axi = compare_surface(self.results, self.labels, self.values)
			if save_fig_basename:
				fig.savefig(join(AUTO_IMAGES_DIR, '{0:s}_surf.png'.format(save_fig_basename)))
		else:
			print 'There are more than two parameters to compare; no visualization options.'
		if self.dims:
			self.print_top(topprint)
		show()
		return self.results


def store_results(filepath, logloss, accuracy, duration):
	with open(filepath, 'w+') as fh:
		fh.write(dumps({'logloss': logloss, 'accuracy': accuracy, 'duration': duration}, indent = 4, sort_keys = True))

def load_results(filepath):
	"""
		:return: (logloss, accuracy, duration) tuple
	"""
	with open(filepath, 'r') as fh:
		d = loads(fh.read())
	return d['logloss'], d['accuracy'], d['duration']


def is_number(obj):
	return isinstance(obj, float) or isinstance(obj, int)


def is_nonstr_iterable(obj):
	"""
		http://stackoverflow.com/questions/19943654/type-checking-an-iterable-type-that-is-not-a-string
	"""
	return not isinstance(obj, str) and isinstance(obj, Iterable)

def params_name(params, prefix):
	params = OrderedDict(sorted(params.items()))
	return (
		sha1(prefix + '_'.join('{0:s}-{1:}'.format(key, val) for key, val in params.items())).hexdigest(),
		', '.join('{0:s} = {1:}'.format(key, val) for key, val in params.items()),
	)
