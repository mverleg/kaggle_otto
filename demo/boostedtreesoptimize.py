# -*- coding: utf-8 -*-
"""
Created on Fri May 01 15:31:03 2015

@author: Fenno
"""

from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer
from validation.optimize_parallel import ParallelGridOptimizer
from boosted_trees.boostedtrees import boostedTrees


max_iterations = 200
min_child_weight = []
step_size = []
max_depth = []
min_loss_reduction = [0, 0.01, 0.05, 0.1, 0.5]
verbose = 0
class_weight = [None, "auto"]

outlier_frac = [False, 0.03, 0.06, 0.01]
outlier_method = ['EE', 'OCSVM']
rescale_pred = [False,True]

baseparams = { "verbose": verbose, "max_iterations": max_iterations }
testparams = baseparams.copy()
testparams.update({"min_loss_reduction", min_loss_reduction
#                   "outlier_method": outlier_method,
#                   "rescale_pred": rescale_pred,
#                   "sample_weight": sample_weight})

train_data, true_classes, _ = get_training_data()    
validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)

#The parallelized code
optimizer = ParallelGridOptimizer(boostedTrees, validator, use_caching=False, process_count=1, **testparams)
optimizer.readygo(only_show_top=False)

"""
optimizer = GridOptimizer(validator, **testparams)
for params, train, classes, test in optimizer.yield_batches():
    prediction = randomForest(train, classes, test, **params)
    optimizer.register_results(prediction)
optimizer.print_top(12)

"""
