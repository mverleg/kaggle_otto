
"""
	This class lets you do cross validation as simply as possible (or as simple as I could come up with).

	See discussion on issue https://gitlab.science.ru.nl/maverickZ/kaggle-otto/issues/6
"""

from random import Random
from sys import stdout
from time import time
from numpy import array, setdiff1d
from subprocess import check_output
from validation.score import calc_logloss, calc_accuracy


class SampleCrossValidator():
	"""
		Facilitates cross validation by providing series of train and test data on which to run your own code. The results can be returned to this class to get performance metrics. Brief example in demo/test_crossvalidate.py .
	"""

	def __init__(self, data, true_classes, test_frac = 0.3, use_data_frac = None, show = True, seed = 4242):
		"""
			Construct a validator instance, binding data and parameters.

			:param data: Array with all the training data (no need to shuffle) with sample rows and feature columns.
			:param true_classes: Array with the true class label integers.
			:param test_frac: Optionally, the fraction of the used data to assign for testing (the rest being training).
			:param use_data_frac: Optionally, the fraction of the total data to include in test and training.
			:param show: Whether to print output each time a probability is added; defaults to True.
			:param seed: A fixed seed for sampling, so that the same data yields the same results consistently. Should probably not be changed.
		"""
		assert data.shape[0] == true_classes.shape[0], 'There should be a true class for each sample ({0:d} vs {1:d}).'.format(data.shape[1], true_classes.shape[0])
		assert 0 < test_frac < 1
		assert 0 < use_data_frac < 1 + 1e-6 or use_data_frac is None
		self.data = data
		self.true_classes = true_classes
		self.test_frac = test_frac
		self.use_data_frac = use_data_frac
		self.show = bool(show)
		self.random = Random(seed)
		self.samples = []
		self.results = []
		self.yield_time = None
		self.total_data_count = self.data.shape[0]
		self.use_data_count = int(self.total_data_count * use_data_frac) if use_data_frac else self.total_data_count
		self.test_count = int(test_frac * self.use_data_count)

	def get_cross_validation_set(self):
		"""
			Get one pair of shuffled train and test data. Intended for internal use.

			:return: train_data, train_classes, test_data, test_classes
		"""
		""" Get the indices of the data being used (sampled randomly). """
		if self.use_data_frac:
			use_indices = array(self.random.sample(range(self.total_data_count), self.use_data_count))
		else:
			use_indices = array(range(self.total_data_count))
		""" Get the indices of the testing data randomly (subset of the data being used). """
		test_indices = array(self.random.sample(use_indices, self.test_count))
		""" The set difference is a series of n 'in' operations, which are O(1) hashmap lookups. So it scales well. """
		train_indices = setdiff1d(use_indices, test_indices)
		self.random.shuffle(train_indices)
		""" Return the information for predicting the test data. """
		return self.data[train_indices, :], self.true_classes[train_indices], self.data[test_indices, :], self.true_classes[test_indices]

	def yield_cross_validation_sets(self, rounds = 3):
		"""
			Yields each of the sets of cross validation data.

			:param rounds: How many rounds of cross validation to perform.
			:return: An iterator with (train_data, train_classes, test_data) tuple on each iteration.
		"""
		assert len(self.samples) == 0, 'This {0:s} already has samples; create a new one if you want to cross-validate again.'.format(self.__class__.__name__)
		for round in range(rounds):
			""" Store the test indices for validation later (I hope train won't be needed). """
			train_data, train_classes, test_data, test_classes = self.get_cross_validation_set()
			self.samples.append(test_classes)
			self.yield_time = time()
			yield train_data, train_classes, test_data

	def add_prediction(self, prediction):
		"""
			Register a classification result for scoring.

			:param prediction: SxC array with predicted probabilities, with each row corresponding to a test data sample and each column corresponding to a class.
			:return: (logloss, accuracy) tuple of floats
		"""
		duration = time() - self.yield_time
		#assert prediction.shape[1] == NCLASSES, 'There should be a probability for each class.'
		assert len(self.results) < len(self.samples), 'There is already a prediction for each sample generated.'
		test_classes = self.samples[len(self.results)]
		logloss = calc_logloss(prediction, test_classes)
		accuracy = calc_accuracy(prediction, test_classes)
		if self.show and not len(self.results):
			stdout.write('  #   loss   accuracy  time\n')
		self.results.append((logloss, accuracy, duration,))
		if self.show:
			stdout.write('{0:-3d}  {1:6.3f}  {2:5.2f}%  {3:6.3f}s\n'.format(len(self.results), logloss, 100 * accuracy, duration))
		return logloss, accuracy

	def get_results(self):
		"""
			:return: List of arrays [logloss, accuracy, duration] with a value for each iteration.
		"""
		return [array(li) for li in zip(*self.results)]

	def print_results(self, output_handle = stdout):
		"""
			Print some results to output_handle (console by default).

			The current git hash is included so that the result can hopefully be reproduced by going back to that commit.
		"""
		logloss, accuracy, duration = self.get_results()
		output_handle.write('*cross validation results*\n')
		output_handle.write('code version  {0:s}\n'.format(check_output(['git', 'rev-parse','HEAD']).rstrip()))
		output_handle.write('repetitions   {0:d}\n'.format(len(self.results)))
		output_handle.write('training #    {0:d}\n'.format(self.use_data_count - self.test_count))
		output_handle.write('testing #     {0:d}\n'.format(self.test_count))
		output_handle.write('                  mean       min       max\n'.format(self.test_count))
		output_handle.write('logloss       {0:8.4f}  {1:8.4f}  {2:8.4f}\n'.format(logloss.mean(), logloss.min(), logloss.max()))
		output_handle.write('accuracy      {0:7.3f}%  {1:7.3f}%  {2:7.3f}%\n'.format(100 * accuracy.mean(), 100 * accuracy.min(), 100 * accuracy.max()))
		output_handle.write('duration      {0:7.4f}s  {1:7.4f}s  {2:7.4f}s\n'.format(duration.mean(), duration.min(), duration.max()))
		if logloss.mean() < 0.4:
			output_handle.write('amazing results!!\n')
		elif logloss.mean() < 0.435:
			output_handle.write('good results!\n')
		elif logloss.mean() < 0.451:
			output_handle.write('pretty good\n')
		elif logloss.mean() < 0.476:
			output_handle.write('okay results\n')
		elif logloss.mean() < 0.55:
			output_handle.write('not that bad\n')
		elif logloss.mean() < 1:
			output_handle.write('room for improvement\n')
		elif logloss.mean() < 2.19722:
			output_handle.write('meh...\n')
		else:
			output_handle.write('not good, sorry\n')



