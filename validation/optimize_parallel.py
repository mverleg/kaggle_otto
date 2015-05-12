
"""
	General code to optimize parameters to a model.
"""

from copy import deepcopy
from functools import partial
from collections import Iterable
from multiprocessing import cpu_count, Pool
from os.path import join, basename, splitext
from sys import stderr, modules
from datetime import datetime
from numpy import prod, unravel_index
from settings import OPTIMIZE_RESULTS_DIR, VERBOSITY, LOGS_DIR
from validation.optimize import GridOptimizer, load_results, params_name, store_results


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
		super(ParallelGridOptimizer, self).__init__(validator, use_caching = use_caching, prefix = prefix, **params)
		self.train_test_func = train_test_func
		self.process_count = process_count
		self.warning_shown = False

	def yield_batches(self, *args, **kwargs):
		if not self.warning_shown:
			stderr.write('To get the benefit of parallellization, you should use .readygo() instead .yield_batches(). The code structure is quite different, consult some instructions.\n')
			self.warning_shown = True
		super(ParallelGridOptimizer, self).yield_batches(*args, **kwargs)

	def readygo(self, print_current_parameters = VERBOSITY, topprint = 12, save_fig_basename = None, log_name = None, only_show_top = False):
		"""
			Start executing jobs in parallel.
		"""
		""" Some setup stuff """
		if log_name is None:
			log_name = '{0:s}.log'.format(splitext(basename(getattr(modules['__main__'], '__file__', 'optimize.default')))[0])
		make_logfile(log_name)
		""" Create all the parameter sets. """
		all_params = []
		for p in range(prod(self.dims, dtype = int)):
			coord = unravel_index(p, self.dims) if self.dims else tuple()
			params = {self.labels[d]: self.values[d][k] for d, k in enumerate(coord)}
			params.update(self.fixed_params)
			all_params.append(params)
		""" See which have already been calculated. """
		todo_jobs = []
		for index, params in enumerate(all_params):
			for round in range(self.validator.rounds):
				if self.use_caching:
					filename, dispname = params_name(params, self.prefix)
					try:
						""" Try to load cache. """
						results = load_results(join(OPTIMIZE_RESULTS_DIR, filename + 'r{0:d}.result'.format(round)))
					except IOError:
						""" No cache; this one is still to be calculated. """
						todo_jobs.append((index, params, round))
					else:
						""" Cache loaded; handle. """
						self.add_results(index * self.rounds + round, *results)
						log_result(log_name, dispname, *results)
						if print_current_parameters:
							print 'cache: %s, round #%d/%d' % (dispname, round + 1, self.rounds)
				else:
					todo_jobs.append((index, params, round))
		""" Calculate probabilties for the others using subprocesses. """
		func = partial(job_handler,
			train_test_func = self.train_test_func,
			validator = deepcopy(self.validator),
			prefix = self.prefix,
			log_name = log_name,
		)
		pool = Pool(processes = self.process_count)
		job_results = pool.map(func, todo_jobs)
		""" Convert probabilities to scores and store them. """
		for (index, params, round), result in zip(todo_jobs, job_results):
			self.add_results(index * self.rounds + round, *result)
		""" Visualize. """
		if only_show_top:
			self.print_top(topprint = topprint)
			return self
		self.print_plot_results(topprint = topprint, save_fig_basename = save_fig_basename)
		return self


def make_logfile(filename):
	with open(join(LOGS_DIR, filename), 'w+') as fh:
		fh.write('# datetime\tparameters\tloss\taccuracy(%)\tduration(s)\n')


def log_result(filename, param_name, *results):
	result_txt = '{0:s}\t{1:8.5f}\t{2:7.4f}\t{3:8.5f}'.format(param_name, *results)
	with open(join(LOGS_DIR, filename), 'a+') as fh:
		fh.write('{1:s}\t{0:s}\n'.format(result_txt, datetime.now().strftime("%Y-%d-%B %H:%M:%S")))


def job_handler(tup, train_test_func, validator, prefix, log_name):
	index, params, round = tup
	filename, dispname = params_name(params, prefix)
	if VERBOSITY >= 2:
		print 'calculate: %s, round #%d/%d' % (dispname, round + 1, validator.rounds)
	train, classes, test = validator.start_round(round)
	prediction = train_test_func(train, classes, test, **params)
	results = validator.add_prediction(prediction, _force_round = round)
	store_results(join(OPTIMIZE_RESULTS_DIR, filename + 'r{0:d}.result'.format(round)), *results)
	print '{0:s}\t{1:6.3f}\t{2:5.2f}%\t{3:6.3f}s'.format(dispname, *results)
	log_result(log_name, dispname, *results)
	return results


