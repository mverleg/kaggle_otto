
"""
	Create new training data samples.
"""

from matplotlib.pyplot import subplots, show
from numpy import empty, float64, uint16, bincount, array, where, concatenate, vstack, hstack
from settings import TOP_PREDICTIONS, TESTSIZE, NCLASSES, VERBOSITY
from utils.loading import get_training_data, get_testing_data


def expand_from_test(train_data, labels, test_data, confidence = 0.9, predictions_path = TOP_PREDICTIONS):
	if confidence is None:
		return train_data, labels
	if VERBOSITY >= 1:
		print 'adding confident test samples to train data'
	selector = ConfidentTestSelector(predictions_path = predictions_path, prior_sizes = bincount(labels)[1:],
		test_data = test_data, confidence = confidence)
	return selector.add_to_train(train_data, labels)


class ConfidentTestSelector(object):
	"""
		Given high-scoring predictions, get a set of confident predictions from test data, rebalance them to keep prior probabilities, then add them to training data.
	"""

	def __init__(self, predictions_path, prior_sizes, test_data, confidence = 0.9):
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
		self.prior_sizes = prior_sizes
		self.confidence = confidence

	def plot(self):
		"""
			Plot an image of the confident test class balance vs priots. Remember to call show().
		"""
		raw_labels = selector.make_raw_data()[1]
		balanced_labels = selector.make_balanced_data()[1]
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

	def make_raw_data(self):
		"""
			Extract the raw confident test data and labels.
		"""
		filter = self.predictions.max(1) > self.confidence
		data = self.test_data[filter, :]
		labels = self.predictions[filter, :].argmax(1) + 1
		return data, labels

	def make_balanced_data(self):
		"""
			Extract the raw confident test data and labels, trimming it to preserve priors size.
		"""
		data, labels = self.make_raw_data()
		extra_sizes = bincount(labels - 1)
		frac = extra_sizes.astype(float64) / self.prior_sizes
		limits = (min(frac) * self.prior_sizes).astype(uint16)
		select = []
		for cls in range(1, NCLASSES + 1):
			select.append(where(labels == cls)[0][:limits[cls - 1]])
		filter = concatenate(select)
		return data[filter, :], labels[filter]

	def add_to_train(self, trian_data, train_labels):
		extra_data, extra_labels = self.make_balanced_data()
		return vstack((trian_data, extra_data)), concatenate((train_labels, extra_labels))


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
	selector = ConfidentTestSelector(TOP_PREDICTIONS, prior_sizes, get_testing_data()[0], confidence = 0.9)
	extra_data, extra_labels = selector.make_balanced_data()
	bigger_data, bigger_labels = selector.add_to_train(train_data, true_labels)
	print 'increase: {0:.2f}%'.format(float(len(bigger_labels)) / len(true_labels) * 100 - 100)
	selector.plot()
	show()


