# -*- coding: utf-8 -*-
"""
Created on Fri May 01 15:31:03 2015

@author: Fenno
"""

from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer
from validation.optimize_parallel import ParallelGridOptimizer
from svm.svm import svm

verbose = 0
class_weight = [None, "auto"] #None seems slightly better
calibration = [0.05,0.1,0.15,0.2,0.25,2,3,4,5]
calibrationmethod = 'isotonic' #['sigmoid', 'isotonic']
outlier_frac = [False, 0.03, 0.06, 0.01]
outlier_method = ['EE', 'OCSVM']
rescale_pred = True #[False,True]
sample_weight = [None, "inverted"]

C = [0.5,1.0,1.5,2.0,3.0]
kernel = 'rbf' #optimize 1 kernel at a time
degree = 3
gamma = [0.0, 0.5, 10, 50, 93]
coef0 = 0.0
shrinking = [False, True]
tol = [1e-5,1e-4, 1e-3, 1e-2]

baseparams = { "verbose": verbose, "kernel" : kernel }
testparams = baseparams.copy()
#testparams.update({"calibration": calibration, "calibrationmethod": calibrationmethod} )
testparams.update({"C" : C,
                    "degree" : degree,
                    "gamma" : gamma,
                    "coef0" : coef0,
                    #"shrinking" : shrinking,
                    "tol" : tol})

train_data, true_classes, _ = get_training_data()    
validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)

#The parallelized code
optimizer = ParallelGridOptimizer(svm, validator, use_caching=False, process_count=50, **testparams)
optimizer.readygo(only_show_top=False)

"""
optimizer = GridOptimizer(validator, **testparams)
for params, train, classes, test in optimizer.yield_batches():
    prediction = randomForest(train, classes, test, **params)
    optimizer.register_results(prediction)
optimizer.print_top(12)

"""
