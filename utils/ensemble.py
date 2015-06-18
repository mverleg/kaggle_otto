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
from utils.loading import get_testing_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer
from validation.optimize_parallel import ParallelGridOptimizer
from boosted_trees.boostedtrees import boostedTrees
from random_forest.randomforest import randomForest

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
    #print Q, N, C
    #print np.shape(weights)
    if weights is None:
        weights = np.ones((Q,C)) * (1.0 / C)
        
    #weights = np.repeat(weights[:,None,:],N,1)
    #result = np.average(predictionmatrix, axis = 0, weights = weights)
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
    count = 0
    for weights, _, _, test in optimizer.yield_batches(False):
        stackedPredict = stack_predictions(test, Q)
        prediction = mean_ensemble(stackedPredict, weights = weightDict[weights['weights']])
        optimizer.register_results(prediction)
        count += 1
        if (count % 100) == 0:
            print count
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
                              

WW = dict(enumerate(make_weights_list(11, 8)))
probsclone = np.ones((11,9))

def classifierEnsemble(train, trueclasses, test, featurestrain, featurestest, model = None):
    """Given test, a QxNxC matrix, trueclasses, a N-array, and test, a QxMxC matrix
      Where Q is number of models, C is number of classes
      model is an optional function that takes in a NxF trainmatrix, N-array true classes, and MxF test matrix
        and returns a MxC prediction matrix
      by default, xgboost is used
      
      featurestrain and featurestext are the feature matrices
      """    
    if model is None:
        model = lambda tr,cl,te : randomForest(tr,cl,te,n_estimators=400,verbose=1,max_depth=None,
                                  min_samples_split=2,min_samples_leaf=1, max_features = 'sqrt',
                                  class_weight=None,calibration=15,n_jobs = 20,rescale_pred=True)
        #model = lambda tr,cl,te : boostedTrees(tr,cl,te, max_iterations=200,min_child_weight=10,max_depth=50,
        #                          step_size=0.1,min_loss_reduction=0.5,verbose=1,rescale_pred=True)
    
    print np.shape(train), np.shape(trueclasses), np.shape(test), np.shape(featurestrain), np.shape(featurestest)
    Q,N,C = np.shape(train)  
    assert N == len(trueclasses)
    Q2,M,C2 = np.shape(test)
    assert (Q2 == Q) and (C2 == C)
    print trueclasses, trueclasses.dtype
    
    unstacktrain = unstack_predictions(train) #is now a NxQC matrix
    unstacktest = unstack_predictions(test)   #is now a MxQC matrix 
    
    
    print np.shape(unstacktrain), np.shape(unstacktest), np.shape(featurestrain), np.shape(featurestest)
    
    unstacktrain = np.hstack((unstacktrain, featurestrain))
    unstacktest = np.hstack((unstacktest, featurestest))
    
    print np.shape(unstacktrain), np.shape(unstacktest)
      
    return model(unstacktrain, trueclasses, unstacktest)

def train_test(train, classes, test, Q, weights, column):
    global WW
    global probsclone
    stackedPredict = stack_predictions(test, Q)
    probsclone[:,column] = WW[weights]
    return multiclass_mean_ensemble(stackedPredict, probsclone)
    
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
    
    optimizer = ParallelGridOptimizer(train_test_func = train_test, validator = validator,  use_caching = False, 
                                     process_count = 1, Q = Q, column = column, weights = weightDict.keys())
    optimizer.readygo()
    """
    optimizer = GridOptimizer(validator = validator, use_caching = False, weights = weightDict.keys())
    count = 0
    for weights, _, _, test in optimizer.yield_batches(False):
        stackedPredict = stack_predictions(test, Q)
        probsclone[:,column] = weightDict[weights['weights']]
        prediction = multiclass_mean_ensemble(stackedPredict, probsclone)
        optimizer.register_results(prediction)
        count += 1
        if (count % 100) == 0:
            print count 
    """    
    bestweight = weightDict[optimizer.print_top(printtop, True)[0][0]]
    probsclone[:,column] = bestweight
    print probsclone
    return multiclass_grid_ensemble(predictionmatrix, trueclasses, probsclone, column + 1, numinterval, False, printtop, data_frac, rounds)
    #return validator.get_results()
    
    
    
from validation.score import calc_logloss 

def selectParentTournament(fitness):
    num = len(fitness)
    cand1 = np.random.randint(0, num)
    cand2 = np.random.randint(0, num)
    return cand1 if fitness[cand1] < fitness[cand2] else cand2
   

def getChild(Pop, pc, fitness):
    C = np.shape(Pop)[1]
    par1 = selectParentTournament(fitness)
    if np.random.random() > pc:
        return Pop[par1,:]    
    par2 = selectParentTournament(fitness)
    crossoverpoint = np.random.randint(1,C)
    return np.concatenate([Pop[par1,:crossoverpoint], Pop[par2,crossoverpoint:]])
    
def mutate(p, pm,r):
    return [r() if np.random.rand() < pm else q for q in p]
    
def geneticEnsemble(predictions, trueclasses, precision=20):
    
    Q,N,C = np.shape(predictions)
    
    P = 100 #population size
    pc = 0.7 #crossover probability
    pm = 1.0/C #mutation probability
    maxiterations = 1000
    
    if precision is None:
        r = lambda : np.random.rand()
    else:
        r = lambda : np.random.randint(precision)
    
    Population = np.array([[r() for j in range(Q)] for i in range(P)])  
    print Population[5,:]
    iterations = 1
    bestFitness = [1000, None]
    fitnessTracer = np.zeros(maxiterations)
    
    while iterations < maxiterations:
        if iterations % 10 == 0:
            print 'At iteration ', iterations
        fitness = np.array([calc_logloss(mean_ensemble(predictions, Population[i,:]), trueclasses) for i in range(P)])
        fitnessTracer[iterations] = fitness.min()
        if fitness.min() < bestFitness[0]:
            bestFitness = [fitness.min(), Population[np.argmin(fitness),:]]
        
        Population = np.array([getChild(Population, pc, fitness) for i in range(P)])
        Population = np.array([mutate(p, pm, r) for p in Population])
        iterations += 1
    
    print fitnessTracer
    return bestFitness  
    
if __name__ == '__main__':
    
    forest1 = np.load('data/predictions/forest1.npy')
    gradient1 = np.load('data/predictions/gradient1.npy')
    boosted1 = np.load('data/predictions/boosted1.npy')
    forest2 = np.load('data/predictions/forest2.npy')
    gradient2 = np.load('data/predictions/gradient2.npy')
    boosted2 = np.load('data/predictions/boosted2.npy')    
    forest3 = np.load('data/predictions/forest3.npy')
    gradient3 = np.load('data/predictions/gradient3.npy')
    boosted3 = np.load('data/predictions/boosted3.npy')
    
    svmprediction = np.load('data/predictions/svmprediction.npy')
    nnprediction = np.load('data/predictions/nnpred.npy')
    knn4 = np.load('data/predictions/knnprediction4.npy')
    knn32 = np.load('data/predictions/knnprediction32.npy')
    knn256 = np.load('data/predictions/knnprediction256.npy')
    treeboost = np.genfromtxt('data/predictions/treeboosting.csv', delimiter=',')       
    goodnn = np.load('data/predictions/goodnn.npy')    
    
    trueclasses = np.load('data/testclas.npy')

    models = (forest1, gradient1, boosted1, forest2, gradient2, boosted2, forest3, gradient3, boosted3, 
              svmprediction, nnprediction, knn4, knn32, knn256, treeboost)
              
    print np.shape(knn4), np.shape(svmprediction)
    p = np.hstack(models)
    p = stack_predictions(p, len(models))
    
    
    print "p", np.shape(p)
    
    #bestfitness =  geneticEnsemble(p, trueclasses, 40)
    #print bestfitness    
    #import sys
    #sys.exit(0)
    
    
    foresttest1 = np.load('data/predictions/forest1100.npy')
    gradienttest1 = np.load('data/predictions/gradient1100.npy')
    boostedtest1 = np.load('data/predictions/boosted1100.npy')
    foresttest2 = np.load('data/predictions/forest2100.npy')
    gradienttest2 = np.load('data/predictions/gradient2100.npy')
    boostedtest2 = np.load('data/predictions/boosted2100.npy')
    foresttest3 = np.load('data/predictions/forest3100.npy')
    gradienttest3 = np.load('data/predictions/gradient3100.npy')
    boostedtest3 = np.load('data/predictions/boosted3100.npy')
    
    svmtest = np.load('data/predictions/svmprediction100.npy')
    nntest = np.load('data/predictions/nnpred100.npy') 
    knn4test = np.load('data/predictions/knnprediction4100.npy')
    knn32test = np.load('data/predictions/knnprediction32100.npy')
    knn256test = np.load('data/predictions/knnprediction256100.npy')
    treeboosttest = np.genfromtxt('data/predictions/treeboosting100.csv', delimiter=',')
    goodnntest = np.load('data/predictions/goodnn100.npy')
    
    testmodels = (foresttest1, gradienttest1, boostedtest1, foresttest2, gradienttest2, boostedtest2, foresttest3, gradienttest3, boostedtest3, 
                 svmtest, nntest, knn4test, knn32test, knn256test, treeboosttest)
    
    ptest = np.hstack(testmodels)    
    ptest = stack_predictions(ptest, len(testmodels))
    print "ptest", np.shape(ptest)
    
      
    weights = [ 1, 24, 28,  0,  0, 31, 38,  1, 34, 18,  0,  0, 4]
    #weights = bestfitness[1]
    
    trainfeat = np.load('data/testmat.npy')
    testfeat, _ = get_testing_data()
    
    classi = classifierEnsemble(p, trueclasses, ptest, trainfeat[:,1:], testfeat)
    #mean = mean_ensemble(ptest, weights)
    
    from utils.ioutil import makeSubmission
    import os
    if os.path.isfile('classsub.csv'):
        os.remove('classsub.csv')
    makeSubmission(classi, 'classsub.csv')
    #if os.path.isfile('meansub.csv'):
    #    os.remove('meansub.csv')
    #makeSubmission(mean, 'meansub.csv')
    
    print "Done making submission"
