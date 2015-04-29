
"""
	General code to optimize parameters to a model.
"""

from hashlib import sha1
from collections import Iterable, OrderedDict
from json import dumps, loads
from multiprocessing import cpu_count, Pool
from os.path import join
from sys import stdout, stderr
from matplotlib.pyplot import show
from numpy import zeros, prod, float64, unravel_index, ravel_multi_index, where
from settings import OPTIMIZE_RESULTS_DIR, VERBOSITY, AUTO_IMAGES_DIR
from validation.crossvalidate import Validator
from validation.optimize import GridOptimizer, load_results
from validation.views import compare_bars, compare_plot, compare_surface


def is_nonstr_iterable(obj):
	"""
		http://stackoverflow.com/questions/19943654/type-checking-an-iterable-type-that-is-not-a-string
	"""
	return not isinstance(obj, str) and isinstance(obj, Iterable)


class ParallelGridOptimizer(GridOptimizer):

	def __init__(self, train_test_func, validator, use_caching = True, prefix = None, process_count = max(cpu_count() - 1, 1), **params):
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
		super(ParallelGridOptimizer, self).__init__(validator, use_caching = True, prefix = None, **params)
		self.train_test_func = train_test_func
		self.process_count = process_count
		self.warning_shown = False

	def yield_batches(self, *args, **kwargs):
		if not self.warning_shown:
			stderr.write('To get the benefit of parallellization, you should use .readygo() instead .yield_batches(). The code structure is quite different, consult some instructions.\n')
			self.warning_shown = True
		super(ParallelGridOptimizer, self).yield_batches(*args, **kwargs)

	def readygo(self, print_current_parameters = VERBOSITY):
		"""
			Start executing jobs in parallel.
		"""
		job_args = []
		for p in range(prod(self.dims, dtype = int)):
			""" Every combination of parameters. """
			coord = unravel_index(p, self.dims) if self.dims else tuple()
			params = {self.labels[d]: self.values[d][k] for d, k in enumerate(coord)}
			params.update(self.fixed_params)
			job_args.append(params)

			filename, dispname = self.params_name(params)
			if print_current_parameters:
				print 'calculating {0:d} rounds for parameters {1:s}'.format(self.rounds, dispname)
			for round in range(self.rounds):
				job_args.append(self.get_single_batch(params, round, dispname))
		pool = Pool(processes = self.process_count)
		job_results = pool.map()


def job_handler(use_caching, job_params, validator_params):
	#todo: validator = ...
	if use_caching:
		try:
			""" Try to load cache. """
			res = load_results(join(OPTIMIZE_RESULTS_DIR, filename + 'r{0:d}.result'.format(round)))
		except IOError as err:
			""" No cache; yield the data (storage happens elsewhere). """
			yield self.get_single_batch(params, round, dispname)
		else:
			""" Cache loaded; handle. """
			self.add_results(*res)
			if VERBOSITY >= 2:
				print 'cache: %s, round #%d/%d' % (dispname, round + 1, self.rounds)
	else:
		raise NotImplementedError('todo')


