# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:58:22 2015

@author: Fenno
"""

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from math import floor
from numpy import ones, shape, bincount
from nnet.prepare import LogTransform
from random_forest.randomforest import randomForest
from utils.features import PositiveSparseRowFeatureGenerator, DistanceFeatureGenerator
from utils.features import PositiveSparseFeatureGenerator
from utils.preprocess import obtain_class_weights, equalize_class_sizes
from utils.outliers import filter_data
from utils.postprocess import rescale_prior
from numpy import sqrt

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
         sample_weight = None,
         rescale = True):
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
    :param rescale: both the training and testing data are taken square root of, rescaled to unit variance, and moved to interval [0,1]
    """
    if outlier_frac > 0:
        train, labels = filter_data(train, labels, cut_outlier_frac = outlier_frac, method = outlier_method)  # remove ourliers
    if isinstance(sample_weight, str):
       sample_weight = obtain_class_weights(labels, sample_weight)
       
    if rescale: #take square root, rescale variance to unit, rescale to [0,1]
        #this should preserve sparsity of matrix
        train = sqrt(train)
        test = sqrt(test)
        scaler = StandardScaler(with_mean=False, with_std = True, copy = True)
        train = scaler.fit_transform(train)        
        scaler = StandardScaler(with_mean=False, with_std = True, copy = True)
        test = scaler.fit_transform(test)
        scaler = MinMaxScaler()
        train = scaler.fit_transform(train)  
        scaler = MinMaxScaler()
        test = scaler.fit_transform(test)        
        
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

    from utils.loading import get_training_data, get_preproc_data
    from validation.crossvalidate import SampleCrossValidator

    train_data, true_classes, _ = get_training_data()
    train, labels, test = get_preproc_data(Pipeline([
	('row', PositiveSparseRowFeatureGenerator()),
	('gen23', PositiveSparseFeatureGenerator(difficult_classes = (2, 3), extra_features = 40)),
	('gen234', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 40)),
	('gen19', PositiveSparseFeatureGenerator(difficult_classes = (1, 9), extra_features = 63)),
	('log', LogTransform()), # log should be after integer feats but before dist
	('distp31', DistanceFeatureGenerator(n_neighbors = 3, distance_p = 1)),
	('distp52', DistanceFeatureGenerator(n_neighbors = 5, distance_p = 2)),
	('scale03', MinMaxScaler(feature_range = (0, 3))), # scale should apply to int and float feats
]), expand_confidence = 0.94)
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
