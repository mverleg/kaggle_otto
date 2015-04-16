
"""
	Code to generate random probabiltities for testing purposes.
"""


from numpy import ones, zeros
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
