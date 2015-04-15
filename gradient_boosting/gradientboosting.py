# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:58:22 2015

@author: Fenno
"""

import sklearn.ensemble as ske

def gradientBoosting(train, labels, test, n_estimators = 50, max_depth = 5, verbose = 0):
    """
    Trains a model by giving it a feature matrix, as well as the labels (the ground truth)
    then using that model, predicts the given test samples
    output is 9 probabilities, one for each class
    """
    model = ske.GradientBoostingClassifier(n_estimators=n_estimators, max_depth = max_depth, verbose = verbose).fit(train, labels) 
    return model.predict_proba(train)
   

if __name__ == '__main__':
    #Dirty hack to use relative path in main file
    import sys
    sys.path.insert(0, '../io')
    import ioutil
    test = ioutil.readTest('../data/test.csv')
    train, labels = ioutil.readTrain('../data/train.csv')
    probs = gradientBoosting(train, labels, test, n_estimators = 10, verbose = 1)    
    ioutil.makeSubmission(probs, '../result.csv')