# -*- coding: utf-8 -*-
"""
Created on Tue Jun 09 11:32:05 2015

@author: Fenno
"""

import numpy as np
from utils.loading import get_training_data

def combinePredictions(predictions, subpredictions, subclasses = [2,3], copyPred = False):
    """
    Takes a Nx9 prediction matrix, and a NxK prediction matrices of a subclassifier
    the subclassifier classifies only the classes in the subclasses list
    then, it combines the two predictions, and returns the new prediction matrix
    copyPred means that the prediction matrix will be copied, which costs more memory
    by default, the Nx9 prediction matrix is overwritten
    """
    assert len(subclasses) == np.shape(subpredictions)[1]
    assert np.shape(subpredictions)[1] < np.shape(predictions)[1]
    assert np.shape(subpredictions)[0] == np.shape(predictions)[0]
    
    if copyPred:
        predictions = np.copy(predictions)
    
    subsums = np.sum(predictions[:,subclasses], 1)    
    predictions[:,subclasses] = np.vstack([subpredictions[:,i]*subsums for i in range(len(subclasses))]).T
    return predictions
    
def getSubClassifierData(subclasses = [2,3], train_data = None, true_classes = None):
   """Gets training data for classification, from only the given classes
   Either filter existing data, or load the default training data, and filter that.
   If either train_data or true_classes is None, the data will be loaded
       using get_training_data() from utils.loading
   """
   if (train_data is None) or (true_classes is None):
       train_data, true_classes, _ = get_training_data()  
   
   assert len(true_classes) == np.shape(train_data)[0]
   
   validsample = np.array([x in subclasses for x in true_classes])
   return train_data[validsample,:], true_classes[validsample,:]
   
if __name__ == "__main__":
    
    
    from random_forest.randomforest import randomForest
    from validation.crossvalidate import SampleCrossValidator
    subclasses = [2,3]    
    
    train_data, true_classes, _ = get_training_data()
    validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)
    for train, classes, test in validator.yield_cross_validation_sets():
        prediction = randomForest(train,classes,test,n_estimators=300,verbose=1,max_depth=None,
                                  min_samples_split=2,min_samples_leaf=1,
                                  class_weight=None,calibration=10,n_jobs = -1,rescale_pred=True)
                                  
        #subtrain,subclasses = getSubClassifierData(subclasses, train, classes)
        #subprediction = randomForest(subtrain, subclasses,test,n_estimators=300,verbose=1,max_depth=None,
        #                          min_samples_split=2,min_samples_leaf=1,
        #                          class_weight=None,calibration=10,n_jobs = -1,rescale_pred=True)
        #prediction = combinePredictions(prediction, subprediction, subclasses)        
              
        validator.add_prediction(prediction)
    validator.print_results()

    