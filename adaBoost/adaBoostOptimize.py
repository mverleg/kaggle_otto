# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 17:02:36 2015

@author: andruta
"""

from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer
from validation.optimize_parallel import ParallelGridOptimizer
from adaBoost import adaBoost
#lol, 3 different ways of naming variables in one line :P

#could tru decisionTreeClassifier also?
base_estimator = [None]
n_estimators = [50, 100, 150, 300]
learning_rate = [1.0, 0.5, 1.5, 2]
algorithm = ['SAMME', 'SAMME.R']
random_state = None
calibration = [0, 0.1]
calibrationmethod = 'sigmoid', 'isotonic'
sample_weight=None,
verbose=1,
outlier_frac=0.0,
outlier_method='EE',
rescale_pred=False


baseparams = { "verbose" : verbose, "n_estimators" : n_estimators,
              "random_state": random_state, "sample_weight":sample_weight,
              "verbose":verbose,  "outlier_frac" : outlier_frac,
              "outlier_method":outlier_method,"rescale_pred": rescale_pred} 

testparams = baseparams.copy()
testparams.update(  {"n_estimators" : n_estimators, "learning_rate" :  learning_rate,
                     "algorithm" : algorithm, "calibration" :  calibration,
                     "calibrationmethod" : calibrationmethod} )

train_data, true_classes, _ = get_training_data()    
validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)

#The parallelized code
"""
optimizer = ParallelGridOptimizer(adaBoost, validator, process_count = 4, **testparams)
optimizer.readygo()
"""

"""
optimizer = ParallelGridOptimizer(adaBoost, validator, process_count = 4, **testparams)

"""

#The non parallelized-code

optimizer = GridOptimizer(validator, **testparams)
for params, train, classes, test in optimizer.yield_batches():
    prediction = adaBoost(train, classes, test, **params)
    optimizer.register_results(prediction)
optimizer.print_plot_results()
