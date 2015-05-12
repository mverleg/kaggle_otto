"""
    Methods to combine several predictions into one
    Typically, the input are the prediction matrices, the true classes, and sometimes the feature matrix as well.
    Then, an ensemble classifier will be trained, using the model predictions as input
    Using this model, you can use the ensemble classifier to classify future samples, given predictions from models.
"""

import numpy as np
import sklearn.linear_model as skl
from utils.arrayutils import stack_predictions, unstack_predictions
from utils.normalize import normalize_probabilities
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
    
def multiclass_mean_ensemble(predictionmatrix, weights = None):
    """
    Computes an ensemble of different models, by taking their mean
    input is a QxNx9 matrix, which is prediction matrices from Q different models
    the weight matrix is a Qx9 matrix, with a weight for every model-class pair
    clearly, the columns should add up to 1
    by default, everything is given equal weight
    """    
    Q, N, C = np.shape(predictionmatrix)
    if weights is None:
        weights = np.ones((Q,C)) * (1.0 / C)
    result = np.empty((N, C))
    for i in range(C):
        result[:,i] = mean_ensemble(predictionmatrix[:,:,i], weights[:,i])
    return normalize_probabilities(result)
    
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
        
    
def grid_ensemble(predictionmatrix, trueclasses, numinterval = None, printWeights = True, printtop = 20, data_frac = 1.0, rounds = 1):
    """
    Does a grid search to find good weights for the ensemble
    predictionmatrix is a QxNxC matrix, where Q is the number of models, N is the number of samples, C is the number of classes
    The parameters for the crossvalidator are fixed because taking a weighted average does not need training and is deterministic
    numinterval is the number of possible weights to consider, evenly spaced between 0 and 1.
    important: numinterval is a postive integer, not a float!
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
    validator = SampleCrossValidator(unstackpredict, trueclasses, rounds=rounds, test_frac=data_frac, use_data_frac=data_frac)
    optimizer = GridOptimizer(validator = validator, use_caching = False, weights = weightDict.keys())
    for weights, _, _, test in optimizer.yield_batches(False):
        stackedPredict = stack_predictions(test, Q)
        prediction = mean_ensemble(stackedPredict, weights = weightDict[weights['weights']])
        optimizer.register_results(prediction)
    print optimizer.print_top(printtop, True)
    #return validator.get_results()
    
def linear_reg_ensemble(predictionmatrix, trueclasses, testmatrix):
    """
    Creates predicts every individual class using a seperate linear regression classifier.
    The inputs are the predictions for every class from every model
    """
    unstackPrediction = unstack_predictions(predictionmatrix)
    unstackTest = unstack_predictions(testmatrix)
    Q, N, C = np.shape(testmatrix)
    result = np.empty((C,N))
    print "Now starting linear regression"
    for i in range(C):
        print "Now starting class " + str(i) 
        model = skl.LinearRegression()
        model.fit(unstackPrediction, trueclasses == (i+1))
        print "Done fitting class " + str(i) + ", now predicting"
        result[i,:] = model.predict(unstackTest)
    return result.T
    
def combineFeatures(a, b):
    #Given NxA matrix and NxB matrix, we want NxAB matrix by pairwise multiplication of columns
    N, A = np.shape(a)
    M, B = np.shape(b)
    assert N == M, (N, M)
    return np.array([np.hstack(np.dot(a[i,:][:,None], b[i,:][None,:])) for i in range(N)])    
    
def fwls_ensemble(predictionmatrix,  trueclasses, testmatrix, predictfeatures, testfeatures):
    """
    Given N training samples, T testing samples, Q models, C classes, F features:
    predictionmatrix: QxNxC matrix
    predictfeatures: NxF matrix
    trueclasses: N array
    testmatrix: QxTxC matrix
    testfeatures: TxF matrix
    Classifies the test samples using FWLS ensemble
    for reference, see: http://arxiv.org/pdf/0911.0460v2.pdf
    """
    Q, _, _ = np.shape(predictionmatrix) 
    trainMatrixFull = combineFeatures(unstack_predictions(predictionmatrix), predictfeatures)
    testMatrixFull = combineFeatures(unstack_predictions(testmatrix), testfeatures)
    print "trainMatrixFull", np.shape(trainMatrixFull)
    print "testMatrixFull", np.shape(testMatrixFull)
    return linear_reg_ensemble(stack_predictions(trainMatrixFull, Q), 
                              trueclasses, 
                              stack_predictions(testMatrixFull, Q))
    
def multiclass_grid_ensemble(predictionmatrix, trueclasses, probssofar = None, column = 0, numinterval = None, printWeights = False, printtop = 20, data_frac = 1.0, rounds = 1):
    Q, _,C = np.shape(predictionmatrix)
    
    if column == C:
        return probssofar
    
    if numinterval is None:
        numinterval = Q + 1
    weightDict = dict(enumerate(make_weights_list(Q, numinterval)))
    
    if printWeights:
        print "The key-value pairs for all possible combinations of weights:"
        for k, v in weightDict.iteritems():
            print str(k) + ": [" + ', '.join([str(e) for e in v]) + "]"
    if probssofar is None:
        probssofar = np.ones((Q,C)) * (1.0 / C)
    probsclone = np.copy(probssofar)
    
    unstackpredict = unstack_predictions(predictionmatrix)
    validator = SampleCrossValidator(unstackpredict, trueclasses, rounds=rounds, test_frac=data_frac, use_data_frac=data_frac)
    optimizer = GridOptimizer(validator = validator, use_caching = False, weights = weightDict.keys())
    count = 0
    for weights, _, _, test in optimizer.yield_batches(False):
        stackedPredict = stack_predictions(test, Q)
        probsclone[:,column] = weightDict[weights['weights']]
        prediction = multiclass_mean_ensemble(stackedPredict, probsclone)
        optimizer.register_results(prediction)
        count += 1
        if (count % 500) == 0:
            print count
        
    bestweight = weightDict[optimizer.print_top(printtop, True)[0][0]]
    probsclone[:,column] = bestweight
    print probsclone
    return multiclass_grid_ensemble(predictionmatrix, trueclasses, probsclone, column + 1, numinterval, False, printtop, data_frac, rounds)
    #return validator.get_results()
    
if __name__ == '__main__':
    
    forest1 = np.load('data/predictions/forest1.npy')
    gradient1 = np.load('data/predictions/gradient1.npy')
    boosted1 = np.load('data/predictions/boosted1.npy')
    forest2 = np.load('data/predictions/forest2.npy')
    gradient2 = np.load('data/predictions/gradient2.npy')
    boosted2 = np.load('data/predictions/boosted2.npy')
    #svmprediction = np.load('data/svmprediction.npy')
    trueclasses = np.load('data/testclas.npy')    

    models = (forest1, gradient1, boosted1, forest2, gradient2, boosted2)
    p = np.hstack(models)
    p = stack_predictions(p, len(models))
    
    print "p", np.shape(p)
    #weightmatrix = multiclass_grid_ensemble(p, trueclasses, numinterval = 11, printtop = 5)
    #print weightmatrix
    #print multiclass_grid_ensemble(p, trueclasses, probssofar = weightmatrix, numinterval = 11, printtop = 1)
    #best weights found: 
    bestweights = np.array([[ 0.4,  0.2,  0.2,  0.3,  0.,   0.2,  0.,   0.1,  0. ],
                             [ 0.1,  0.1,  0.2,  0.2,  0.1,  0.2,  0.5,  0.,   0.2],
                             [ 0.,   0.1,  0.3,  0.2,  0.,   0.2,  0.2,  0.5,  0.6],
                             [ 0.,   0.,   0.3,  0.,   0.,   0.,   0.,   0.,   0. ],
                             [ 0.1,  0.,   0.,   0.,   0.8,  0.1,  0.,   0.,   0. ],
                             [ 0.4,  0.6,  0.,   0.3,  0.1,  0.3,  0.3,  0.4,  0.2]])

    
    
    foresttest1 = np.load('data/predictions/forest1100.npy')
    gradienttest1 = np.load('data/predictions/gradient1100.npy')
    boostedtest1 = np.load('data/predictions/boosted1100.npy')
    foresttest2 = np.load('data/predictions/forest2100.npy')
    gradienttest2 = np.load('data/predictions/gradient2100.npy')
    boostedtest2 = np.load('data/predictions/boosted2100.npy')
    testmodels = (foresttest1, gradienttest1, boostedtest1, foresttest2, gradienttest2, boostedtest2)
    
    ptest = np.hstack(testmodels)    
    ptest = stack_predictions(ptest, len(testmodels))
    print "ptest", np.shape(ptest)
    trainmat = np.load('data/testmat.npy').astype('uint16')
    trainmat = trainmat[:,1:]
    print "trainmat", np.shape(trainmat)
    
    from utils.loading import get_testing_data, get_training_data
    testmat, _ = get_testing_data()
    print "testmat", np.shape(testmat)
    
    linreg = linear_reg_ensemble(p, trueclasses, ptest)
    fwls = fwls_ensemble(p, trueclasses, ptest, trainmat, testmat) 
    grid = multiclass_mean_ensemble(ptest, bestweights)
    
    print "linreg", np.shape(linreg)
    print "fwls", np.shape(fwls)
    print "grid", np.shape(grid)
    
    from utils.ioutil import makeSubmission
    makeSubmission(linreg, 'linregsub.csv')
    makeSubmission(fwls, 'fwlssub.csv')
    makeSubmission(grid, 'gridsub.csv')
