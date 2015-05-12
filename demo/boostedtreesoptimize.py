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




max_iterations = 200 #[100,200,300,400]
min_child_weight = 10 #[0.1, 1, 5, 10, 15, 20]
step_size = 0.1#[0.075,0.1,0.2,0.3] #so far, 0.075-0.1 seems to work best
max_depth = 50#[50,75,100,150]# so far, 50 or 100 seems to work best
min_loss_reduction = 0.5 #[0.1,0.5,1.0,2.0] 0.5 works well
verbose = 1
class_weight = None#None is better by far[None, "auto"]

outlier_frac = [False, 0.03, 0.06, 0.01] #false works best
outlier_method = ['EE', 'OCSVM']
rescale_pred = [False,True] #True works slightly better

baseparams = { "verbose": verbose, "max_iterations": max_iterations }
testparams = baseparams.copy()
testparams.update({ "min_child_weight" : min_child_weight,
                "min_loss_reduction" : min_loss_reduction,
                "step_size" : step_size,
                "max_depth" : max_depth,
                "class_weights" : class_weight,
                "outlier_frac" : outlier_frac,
                "outlier_method" : outlier_method,
                "rescale_pred" : rescale_pred
                  })

train_data, true_classes, _ = get_training_data()    
validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)

#The parallelized code
#optimizer = ParallelGridOptimizer(boostedTrees, validator, use_caching=False, process_count=1, **testparams)
#optimizer.readygo(only_show_top=False)


optimizer = GridOptimizer(validator, use_caching=True, **testparams)
for params, train, classes, test in optimizer.yield_batches():
    print "Now handing out test, waiting for result"
    prediction = boostedTrees(train, classes, test, **params)
    optimizer.register_results(prediction)
optimizer.print_plot_results(12)


