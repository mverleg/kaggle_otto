# -*- coding: utf-8 -*-
"""
Created on Fri May 01 15:31:03 2015

@author: Fenno
"""

from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer
from validation.optimize_parallel import ParallelGridOptimizer
from random_forest.randomforest import randomForest

n_estimators = 200
criterion = ['gini', 'entropy'] #gini seems better (default)
max_features = ['sqrt', 'log2'] #sqrt seems good (default), None is about as good but much slower
max_depth = [None, 30, 35, 40, 75, 100, 200] #at least higher than 20
min_samples_split = [2, 4, 6] #not much difference, 2 (default) seems slightly the best
min_samples_leaf = [1, 2, 3] #1 seems clearly the best (default)
min_weight_fraction_leaf = [0., 0.1, 0.2] #0 seems clearly the best (default)
max_leaf_nodes = [None, 5000, 10000, 50000] #can vary, don't do simultaneously with max_depth
n_jobs = -1
verbose = 1
class_weight = [None, "auto"] #None seems slightly better

calibration = [0.05,0.1,0.15,0.2,0.25,2,3,4,5]
calibrationmethod = 'isotonic' #['sigmoid', 'isotonic']

outlier_frac = [False, 0.03, 0.06, 0.01]
outlier_method = ['EE', 'OCSVM']
rescale_pred = True #[False,True]
sample_weight = [None, "inverted"]

baseparams = { "verbose": verbose, "n_estimators": n_estimators, "n_jobs": n_jobs }
testparams = baseparams.copy()
testparams.update({"calibration": calibration, "calibrationmethod": calibrationmethod} )
testparams.update({"outlier_frac": outlier_frac,
                   "outlier_method": outlier_method,
                   "rescale_pred": rescale_pred,
                   "sample_weight": sample_weight})

train_data, true_classes, _ = get_training_data()    
validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)

#The parallelized code
optimizer = ParallelGridOptimizer(randomForest, validator, use_caching=False, process_count=24, **testparams)
optimizer.readygo(only_show_top=False)

"""
optimizer = GridOptimizer(validator, **testparams)
for params, train, classes, test in optimizer.yield_batches():
    prediction = randomForest(train, classes, test, **params)
    optimizer.register_results(prediction)
optimizer.print_top(12)

"""