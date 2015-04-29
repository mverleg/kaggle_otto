#An example of optimizing the gradient boosting
#The "testparams.update" line contains the variables that need to be evaluated
#if you enter them all, it's like 1000 evaluations, which will take about 240 hours :)

from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer
from gradient_boosting.gradientboosting import gradientBoosting
#lol, 3 different ways of naming variables in one line :P

loss = ['deviance', 'exponential']
learning_rate = [0.05, 0.1, 0.5]
n_estimators = 200
max_depth = [3,5,7]
min_samples_split = [1,2,3]
min_samples_leaf = [1,2,3]
min_weight_fraction_leaf = [0,0.1,0.2]
max_features = ['sqrt','log2',None]
verbose = 1
calibration = [0,0.1]
calibrationmethod = 'sigmoid', 'isotonic'

baseparams = { "verbose" : verbose, "n_estimators" : n_estimators }
testparams = baseparams.copy()
testparams.update(  {"learning_rate" : learning_rate, "max_depth" :  max_depth} )

train_data, true_classes, _ = get_training_data()    
validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)
optimizer = GridOptimizer(validator, **testparams)
for params, train, classes, test in optimizer.yield_batches():
    prediction = gradientBoosting(train, classes, test, **params)
    optimizer.register_results(prediction)
optimizer.print_plot_results()
