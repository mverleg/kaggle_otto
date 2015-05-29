
"""
	Create new training data samples.
"""

from matplotlib.pyplot import subplots, show
from numpy import empty, float64, uint16, bincount, array, where, concatenate, vstack, hstack
from sklearn.base import TransformerMixin, BaseEstimator
from settings import TOP_PREDICTIONS, TESTSIZE, NCLASSES, VERBOSITY
from utils.loading import get_training_data, get_testing_data


def expand_from_test(train_data, labels, test_data, confidence = 0.9, predictions_path = TOP_PREDICTIONS):
	if confidence is None:
		return train_data, labels
	if VERBOSITY >= 1:
		print 'adding confident test samples to train data'
	selector = ConfidentTestSelector(test_data = test_data, confidence = confidence, predictions_path = predictions_path)
	return selector.add_to_train(train_data, labels)


class ConfidentTestSelector(BaseEstimator, TransformerMixin):
	"""
		Given high-scoring predictions, get a set of confident predictions from test data, rebalance them to keep prior probabilities, then add them to training data.

		Note that this may make your scores too optimistic, since you are adding certain data.

		Usage (new style):

			train_data, labels = ConfidentTestSelector(test_data, confidence = 0.9).fit_transform(train_data, labels)

		Update: this doesn't work, since transform cannot transform labels; keep using expand_from_test  #todo
	"""

	def __init__(self, test_data, confidence=0.9, predictions_path = TOP_PREDICTIONS):
		"""
			:param predictions_path: path to the predictions file to use.
			:param: prior sizes: sizes of the classes in train/true data.
			:param test_data: simply the test data which predictions are based on.
			:param confidence: how high should the highest probability be?
			:return:
		"""
		assert 0.5 <= confidence <= 1, 'confidence should be in [0.5 - 1.0]'
		self.test_data = test_data
		self.predictions = load_predictions(predictions_path)
		self.confidence = confidence

	def fit(self, X, y, **fit_params):
		self.prior_sizes = bincount(y)[1:]
		self.data, self.labels = self.make_raw_data()
		extra_sizes = bincount(self.labels - 1)
		frac = extra_sizes.astype(float64) / self.prior_sizes
		self.limits = (min(frac) * self.prior_sizes).astype(uint16)
		return self

	def transform(self, X, y, copy = False):
		""" This is not the correct format; there should be no y. But whatev. """
		extra_data, extra_labels = self.get_extra()
		return vstack((X, extra_data)), concatenate((y, extra_labels)).astype(y.dtype)

	def make_raw_data(self):
		"""
			Extract the raw confident test data and labels.
		"""
		filter = self.predictions.max(1) > self.confidence
		data = self.test_data[filter, :]
		labels = self.predictions[filter, :].argmax(1).astype(uint16) + 1
		return data, labels

	def get_extra(self):
		"""
			Get the extra data (call .fit first).
		"""
		select = []
		for cls in range(1, NCLASSES + 1):
			select.append(where(self.labels == cls)[0][:self.limits[cls - 1]])
		filter = concatenate(select)
		return self.data[filter, :], self.labels[filter]

	def plot(self):
		"""
			Plot an image of the confident test class balance vs priots. Remember to call show().
		"""
		raw_labels = self.make_raw_data()[1]
		balanced_labels = self.get_extra()[1]
		fig, ax1 = subplots()
		ax2 = ax1.twinx()
		x = array(range(1, NCLASSES + 1))
		l1 = ax1.bar(x - 0.3, self.prior_sizes, width = 0.25, color = 'b', align = 'center', label = 'train')
		l2 = ax2.bar(x, bincount(raw_labels - 1), width = 0.25, color = 'r', align = 'center', label = 'confident')
		l3 = ax2.bar(x + 0.3, bincount(balanced_labels - 1), width = 0.25, color = 'g', align = 'center', label = 'rebalanced')
		confident_frac = len(raw_labels) / float(self.predictions.shape[0])
		usable_frac = len(balanced_labels) / float(self.predictions.shape[0])
		ax1.set_title('at >{0:.1f}%, {1:.1f}% reliable, {2:.1f}% usable'.format(self.confidence * 100, confident_frac * 100, usable_frac * 100))
		ax1.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()], loc = 'upper right')
		ax1.set_xticks(x)

	# def make_balanced_data(self):
	# 	"""
	# 		Extract the raw confident test data and labels, trimming it to preserve priors size.
	# 	"""
	# 	data, labels = self.make_raw_data()
	# 	extra_sizes = bincount(labels - 1)
	# 	frac = extra_sizes.astype(float64) / self.prior_sizes
	# 	limits = (min(frac) * self.prior_sizes).astype(uint16)
	# 	select = []
	# 	for cls in range(1, NCLASSES + 1):
	# 		select.append(where(labels == cls)[0][:limits[cls - 1]])
	# 	filter = concatenate(select)
	# 	return data[filter, :], labels[filter]

	def add_to_train(self, train_data, train_labels):
		""" Alias for historic reasons. """
		return self.fit(train_data, train_labels).transform(train_data, train_labels)


def load_predictions(filepath):
	"""
		Load a submission file (without optimization).
	"""
	with open(filepath, 'r') as fh:
		cells = [line.split(',') for line in fh.read().splitlines()]
	data = empty((TESTSIZE, NCLASSES), dtype = float64)
	for k, row in enumerate(cells[1:]):
		data[k, :] = row[1:]
	return data


if __name__ == '__main__':
	train_data, true_labels = get_training_data()[:2]
	prior_sizes = bincount(true_labels)[1:]
	selector = ConfidentTestSelector(get_testing_data()[0], confidence = 0.9)
	bigger_data, bigger_labels = selector.add_to_train(train_data, true_labels)
	print 'increase: {0:.2f}%'.format(float(len(bigger_labels)) / len(true_labels) * 100 - 100)
	selector.plot()
	show()


