
"""
    Functions for postprocessing the predictions after they are made
"""

import numpy as np


def rescale_prior(predictionmatrix, priorprobs):
    """
    Rescales predictions to make them match the prior probability
    :param predictionmatrix: a Nx9 matrix of predictions
    :param priorprobs: a length-9 vector with prior probabilities, does not need to be normalized,
        so can just be a bincount of the true_class vector
    :return: A prediction matrix with rescaled probabilities
    """
    N, C = np.shape(predictionmatrix)
    assert C == np.size(priorprobs)
    priorprobs = priorprobs / np.sum(priorprobs).astype(float)
    averageprediction = np.mean(predictionmatrix, axis = 0)
    factor = priorprobs / averageprediction
    return predictionmatrix * np.repeat(factor[None,:],N,0)
    
