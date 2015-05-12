
"""
	Functions for postprocessing the predictions after they are made
"""

import numpy as np
from numpy import log, sqrt
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from settings import VERBOSITY
from utils.normalize import normalize_probabilities


def rescale_prior(predictionmatrix, priorprobs):
	"""
	Rescales predictions to make them match the prior probability
	:param predictionmatrix: a Nx9 matrix of predictions
	:param priorprobs: a length-9 vector with prior probabilities, does not need to be normalized,
		so can just be a bincount of the true_class vector
	:return: A prediction matrix with rescaled probabilities
	"""
	priorprobs = np.trim_zeros(priorprobs, 'f')
	N, C = np.shape(predictionmatrix)
	assert C == np.size(priorprobs)
	priorprobs = priorprobs / np.sum(priorprobs).astype(float)
	averageprediction = np.mean(predictionmatrix, axis = 0)
	factor = priorprobs / averageprediction
	result = predictionmatrix * np.repeat(factor[None,:],N,0)
	return normalize_probabilities(result)


def scale_to_priors(probabilities, priors):
	"""
		Uses iterative minimization to find the best scale factors to approach priors as close as possible.

		:param probabilities: Normalized probabilities.
		:param priors: Prior probability for each dimension (normalized).

		Simply calculating the scale factor is not optimal because the average probabilities change after renormalization.
	"""
	assert abs(priors.sum() - 1) < 1e-6, 'Prior probabilities not normalized.'
	def prior_mismatch(scale, probs, pris):
		return log(mean_squared_error(normalize_probabilities(scale * probs).mean(0), pris))
	result = minimize(prior_mismatch, x0 = priors / probabilities.mean(0), method = 'BFGS',
		args = (probabilities, priors), options = {'maxiter': 1000})
	if VERBOSITY >= 1:
		print 'scaled to priors; mismatch was {0:.4f} (0 being perfect)'.format(sqrt((result.x - 1)**2))
	return normalize_probabilities(probabilities * result.x)


