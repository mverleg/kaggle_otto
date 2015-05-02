#!/env/bin/python

import lasagne

from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer
from validation.optimize_parallel import ParallelGridOptimizer
from gradient_boosting.gradientboosting import gradientBoosting
#lol, 3 different ways of naming variables in one line :P


#import nnet.mark.optimize_hidden_size

#print "test"
#import sys
#sys.exit("exit")


learning_rate = 0.05 #[0.05,0.075,0.1] #default 0.1
n_estimators = 300
max_depth = 5 #5 works okay, 7 as well, default 5
min_samples_split = 2 #[2,5,7] #really does not matter for [1,2,3]
min_samples_leaf = 5 #[3,5,7] #small difference, but higher seems a little better
min_weight_fraction_leaf = 0 #[0,0.1,0.2] #default 0
max_features = 'sqrt' #['sqrt','log2',None]
verbose = 1
calibration = False#[0,0.1]
calibrationmethod = 'sigmoid' #['sigmoid', 'isotonic']

outlier_frac = False #[False, 0.03, 0.06]
outlier_method = 'EE' #['EE', 'OCSVM']
undersample = False #[False,True] #didn't try, because probably won't work anyway
rescale_pred = False # [False,True] 
sample_weight = None #[None, "inverted"]

baseparams = { "verbose" : verbose, "n_estimators" : n_estimators }
testparams = baseparams.copy()
testparams.update( {"learning_rate" : learning_rate, "max_depth" : max_depth } )
testparams.update(  {"calibration" : calibration, "outlier_frac" : outlier_frac, "max_features" : max_features })
#testparams.update( {"outlier_frac" : outlier_frac, "outlier_method" : outlier_method, "rescale_pred" : rescale_pred, "sample_weight" : sample_weight } )
testparams.update( { "min_weight_fraction_leaf" : min_weight_fraction_leaf } )
testparams.update( { "min_samples_split" : min_samples_split, "min_samples_leaf" : min_samples_leaf } )

train_data, true_classes, _ = get_training_data()
validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)

#Parallelized code
#optimizer = ParallelGridOptimizer(gradientBoosting, validator, use_caching = False, process_count = 27, **testparams)
#optimizer.readygo(only_show_top=True)


optimizer = GridOptimizer(validator, **testparams)
for params, train, classes, test in optimizer.yield_batches():
    prediction = gradientBoosting(train, classes, test, **params)
    optimizer.register_results(prediction)
optimizer.print_top(12)

