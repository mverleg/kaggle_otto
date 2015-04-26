"""
    Methods to combine several predictions into one
    Typically, the input are the prediction matrices, the true classes, and sometimes the feature matrix as well.
    Then, an ensemble classifier will be trained, using the model predictions as input
    Using this model, you can use the ensemble classifier to classify future samples, given predictions from models.
"""

import sys
sys.path.append('..')

import numpy as np
import sklearn.linear_model as skl
from arrayutils import stack_predictions, unstack_predictions
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer

def mean_ensemble(predictionmatrix, weights = None):
    """
    Computes an ensemble of different models, by taking their mean
    input is a QxNx9 matrix, which is prediction matrices from Q different models
    optionally, you can use weights to weigh the models. Weights can be found using grid_ensemble
    by default, all models are given equal weight
    """
    return np.average(predictionmatrix,axis=0, weights = weights)
    
def make_weights_list(numweights, numinterval, totalsum = 1.0):
    """
    Makes a numpy matrix of all possible combinations of weights that sum up to totalsum
    For example, if numweights = 2, numinterval = 3, totalsum = 1.0, it will look like:
    np.array([[0, 1],[0.5, 0.5],[1, 0]])
    The order at which weights are added to the list is not guaranteed, but it is guaranteed
    that every possible combination of weights will appear exactly once.
    numweights: the number of weights
    numinterval: the number of possible weights, evenly spaced between 0 and totalsum
    """
    assert numweights > 0
    if numweights == 1:
        return np.array([[totalsum]])
    result = np.empty((0, numweights))
    firstweight = np.linspace(0,totalsum,numinterval,True)
    for i, x in enumerate(firstweight):
        restweights = make_weights_list(numweights - 1, numinterval - i, totalsum - x)
        N, _ = np.shape(restweights)
        firstcolumn = np.empty((N,1))
        firstcolumn.fill(x)
        result = np.vstack((result, np.hstack((firstcolumn, restweights))))
    return result
        
    
def grid_ensemble(predictionmatrix, trueclasses, numinterval = None, printWeights = True, data_frac = 1.0):
    """
    Does a grid search to find good weights for the ensemble
    predictionmatrix is a QxNxC matrix, where Q is the number of models, N is the number of samples, C is the number of classes
    The parameters for the crossvalidator are fixed because taking a weighted average does not need training and is deterministic
    numinterval is the number of possible weights to consider, evenly spaced between 0 and 1.
    By default, this is Q + 1, where Q is the number of models
    """
    Q, _,_ = np.shape(predictionmatrix)
    if numinterval is None:
        numinterval = Q + 1
    weightDict = dict(enumerate(make_weights_list(Q, numinterval)))
    
    if printWeights:
        print "The key-value pairs for all possible combinations of weights:"
        for k, v in weightDict.iteritems():
            print str(k) + ": [" + ', '.join([str(e) for e in v]) + "]"
    
    unstackpredict = unstack_predictions(predictionmatrix)
    validator = SampleCrossValidator(unstackpredict, trueclasses, rounds=1, test_frac=data_frac, use_data_frac=data_frac)
    optimizer = GridOptimizer(validator = validator, use_caching = False, weights = weightDict.keys())
    for weights, _, _, test in optimizer.yield_batches(False):
        stackedPredict = stack_predictions(test, Q)
        prediction = mean_ensemble(stackedPredict, weights = weightDict[weights['weights']])
        optimizer.register_results(prediction)
    optimizer.print_plot_results(20)
    
#def linear_reg_ensemble(predictionmatrix, trueclasses, testmatrix)


if __name__ == '__main__':
    from demo.fake_testing_probabilities import get_random_probabilities
    N = 100
    Q = 6
    C = 5
    randompredictions = np.array([get_random_probabilities(N, C) for i in range(Q)])
    randomclasslabels = np.random.randint(0,C,N)
    grid_ensemble(randompredictions, randomclasslabels, data_frac = 0.9)
