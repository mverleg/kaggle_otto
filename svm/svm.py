# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:58:22 2015

@author: Fenno
"""

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from math import floor
from numpy import ones, shape, bincount
from utils.preprocess import obtain_class_weights, equalize_class_sizes
from utils.outliers import filter_data
from utils.postprocess import rescale_prior

def svm(train,
         labels,
         test,
         calibration=0.0,
         calibrationmethod='sigmoid',
         C = 1.0,
         kernel = 'rbf',
         degree = 3,
         gamma = 0.0,
         coef0 = 0.0,
         shrinking = True,
         tol = 1e-3,
         verbose = 0,
         outlier_frac=0.0,
         outlier_method='EE',
         rescale_pred=False,
         class_weight=None,
         sample_weight = None):
    """
    Trains a model by giving it a feature matrix, as well as the labels (the ground truth)
    then using that model, predicts the given test samples
    output is 9 probabilities, one for each class
    :param train: The training data, to train the model
    :param labels: The labels of the training data, an array
    :param calibration: How much data to use for calibration. If calibration is 0, no calibration is done.
        The data is simply split, no shuffling is done, so if the data is ordered, shuffle it first!
        If calibration is n > 1, then crossvalidation will be done, using n folds.
    :param verbose: See sklearn documentation
    """
    if outlier_frac > 0:
        train, labels = filter_data(train, labels, cut_outlier_frac = outlier_frac, method = outlier_method)  # remove ourliers
    if isinstance(sample_weight, str):
       sample_weight = obtain_class_weights(labels, sample_weight)
        
    model = SVC(C = C,
               kernel = kernel,
               degree = degree,
               gamma = gamma,
               coef0 = coef0,
               probability = True,
               shrinking = shrinking,
               tol = tol,
               verbose = verbose,
               class_weight = class_weight)

    if calibration == 0.0:
        model.fit(train, labels, sample_weight)
    elif calibration > 1:
        model = CalibratedClassifierCV(model, calibrationmethod, calibration)
        model.fit(train, labels, sample_weight)
    else:
        N = len(labels)
        if sample_weight is None:
            sample_weight = ones(N)
        train_rows = floor((1.0 - calibration) * N)
        model.fit(train[:train_rows, :], labels[:train_rows], sample_weight[:train_rows])
        model = CalibratedClassifierCV(model, calibrationmethod, "prefit")
        model.fit(train[train_rows:, :], labels[train_rows:], sample_weight=sample_weight[train_rows:])

    predictions = model.predict_proba(test)

    if rescale_pred:
        predictions = rescale_prior(predictions, bincount(labels))
    return predictions  


if __name__ == '__main__':

    from utils.loading import get_training_data
    from validation.crossvalidate import SampleCrossValidator

    train_data, true_classes, _ = get_training_data()    
    validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)
    for train, classes, test in validator.yield_cross_validation_sets():
        prediction = randomForest(train,
                                  classes,
                                  test,
                                  n_estimators=200,
                                  max_depth=35,
                                  verbose=1,
                                  class_weight="auto",
                                  calibration=3,
                                  rescale_pred=True)
        validator.add_prediction(prediction)
    validator.print_results()
