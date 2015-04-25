
"""
	Code to generate random probabiltities for testing purposes.
"""

from numpy import ones, zeros, mod, uint16
from numpy.random import uniform
from settings import TRAINSIZE, NCLASSES
from utils.normalize import normalize_probabilities


def get_random_probabilities(sample_count = TRAINSIZE, class_count = NCLASSES):
	return normalize_probabilities(uniform(size = (sample_count, class_count)))


def get_uniform_probabilities(sample_count = TRAINSIZE, class_count = NCLASSES):
	return normalize_probabilities(ones((sample_count, class_count)))


def get_binary_probabilities(sample_count = TRAINSIZE, class_count = NCLASSES):
	predictions = zeros((sample_count, class_count))
	predictions[:, 0] = 1
	return predictions


def get_from_data(data, class_count = NCLASSES):
	"""
		Create probabilities that depend deterministically on the data (but aren't actual predictions).
	"""
	predictions = ones((data.shape[0], class_count))
	chosen_classes = (mod(data.sum(1), class_count) + 1).astype(uint16)
	predictions[range(predictions.shape[0]), chosen_classes - 1] = 10
	return normalize_probabilities(predictions)



