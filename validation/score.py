
"""
	Functions for assessing the performance of different classifiers.
"""

from numpy import array, zeros
from scipy import clip, log
from demo.fake_testing_probabilities import get_random_probabilities, get_uniform_probabilities, get_binary_probabilities
from utils.normalize import check_rows_normalized


def prob_to_cls(probabilities):
	return probabilities.argmax(1) + 1


def calc_logloss(predictions, true_classes, epsilon = 1.e-15):
	"""
		The multiclass logarithmic loss function used in the competition, including the truncation.

		:param predictions: An Nx9 ndarray with prediction probabilities. Rows are samples in the same order as true_classes, and columns correspond to classes (from 1 to 9).
		:param true_classes: A N-element vector with the correct class labels (integers from 1 to 9).
		:return: The logloss (float).

		Information: https://www.kaggle.com/c/otto-group-product-classification-challenge/details/evaluation
		Based on: https://www.kaggle.com/wiki/LogarithmicLoss   but actually changed completely
	"""
	assert check_rows_normalized(predictions), 'The predictions you submitted aren\'t normalized! You can use normalize_probabilities(..). Current norms-1 are in the range {0:.5f}-{1:.5f}'.format((predictions.sum(axis = 1) - 1).min(), (predictions.sum(axis = 1) - 1).max())
	pred = clip(predictions, epsilon, 1 - epsilon)
	predictions_for_true = pred[range(predictions.shape[0]), true_classes - 1]
	return - log(predictions_for_true).sum() / len(true_classes)


def calc_accuracy(predictions, true_classes):
	"""
		The accuracy of the predictions (how many of the most probable classes were correct).

		Arguments are the same as those for logloss.
		:return: The accuracy as a fraction [0-1].
	"""
	return (prob_to_cls(predictions) == true_classes).mean()


if __name__ == '__main__':
	S, C = 21, 9
	true_classes = array([((2 * k) % C) + 1 for k in range(S)])
	predictions = get_random_probabilities(S, C)
	print '          loss    accuracy'
	print 'random:  {0:6.3f}   {1:.3f}'.format(calc_logloss(predictions, true_classes), calc_accuracy(predictions, true_classes))
	predictions = get_uniform_probabilities(S, C)
	print 'uniform: {0:6.3f}   {1:.3f}'.format(calc_logloss(predictions, true_classes), calc_accuracy(predictions, true_classes))
	predictions = get_binary_probabilities(S, C)
	print 'certain: {0:6.3f}   {1:.3f}'.format(calc_logloss(predictions, true_classes), calc_accuracy(predictions, true_classes))
	predictions = zeros((S, C))
	predictions[range(predictions.shape[0]), true_classes - 1] = 1
	print 'correct: {0:6.3f}   {1:.3f}'.format(calc_logloss(predictions, true_classes), calc_accuracy(predictions, true_classes))


