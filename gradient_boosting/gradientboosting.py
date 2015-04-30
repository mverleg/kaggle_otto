# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:58:22 2015

@author: Fenno
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from math import floor
from numpy import ones, shape

def gradientBoosting(train, labels, test, loss = 'deviance', n_estimators = 50, max_depth = 3, learning_rate = 0.1, min_samples_split = 2, min_samples_leaf = 1, min_weight_fraction_leaf = 0, max_features = None, calibration = None, calibrationmethod = 'sigmoid', sample_weight = None,  verbose = 1):
    """
    Trains a model by giving it a feature matrix, as well as the labels (the ground truth)
    then using that model, predicts the given test samples
    output is 9 probabilities, one for each class
    :param train: The training data, to train the model
    :param labels: The labels of the training data, an array
    :param test: the data to predict
    :param n_estimators: See sklearn documentation
    :param max_depth: See sklearn documentation
    :param calibration: How much data to use for calibration. If calibration is False (including 0.0), no calibration is done.
        The data is simply split, no shuffling is done, so if the data is ordered, shuffle it first!
        If calibration is n > 1, then crossvalidation will be done, using n folds.
    :param verbose: See sklearn documentation
    """
    if not calibration: #no calibration is done
        model = GradientBoostingClassifier(loss = loss, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf,max_features = max_features, learning_rate = learning_rate, n_estimators=n_estimators, max_depth = max_depth, verbose = verbose)
        model.fit(train, labels, sample_weight)
        return model.predict_proba(test)
        
    N = len(labels)
    trainrows = floor((1.0 - calibration) * N)
    model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth = max_depth, verbose = verbose)
    
    if calibration > 1:
        calibratedmodel = CalibratedClassifierCV(model, calibrationmethod, calibration)
        calibratedmodel.fit(train, labels, sample_weight)
    else:
        if sample_weight is None:
            sample_weight = ones((len(labels)))
        model.fit(train[:trainrows, :], labels[:trainrows],sample_weight[:trainrows])
        calibratedmodel = CalibratedClassifierCV(model, calibrationmethod, "prefit")
        calibratedmodel.fit(train[trainrows:,:], labels[trainrows:], sample_weight = sample_weight[trainrows:])
    return calibratedmodel.predict_proba(test)    


if __name__ == '__main__':

    from utils.loading import get_training_data
    from validation.crossvalidate import SampleCrossValidator

    train_data, true_classes, _ = get_training_data()    
    validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)
    for train, classes, test in validator.yield_cross_validation_sets():
        prediction = gradientBoosting(train, classes, test, 100, 10, 0.1, verbose = 1)
        #Warning: this will take a while, for faster testing, change the 100 iterations to 1 or something
        validator.add_prediction(prediction)
    validator.print_results()

