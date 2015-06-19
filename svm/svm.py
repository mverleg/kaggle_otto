# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:58:22 2015

@author: Fenno
"""
#best results so far (0.55):C=10, gamma=0.5, kernel='rbf' with no feature selection and no grid search

#lots of these are unnecessary, just tried different stuff, will delete next time
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy import bincount
from utils.preprocess import obtain_class_weights
from utils.outliers import filter_data
from utils.postprocess import rescale_prior
from numpy import sqrt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from math import floor
from numpy import ones

def svm(train,
         labels,
         test,
         C = 10,
         kernel = 'rbf',
         degree = 3,
         gamma = 0.5,
         calibration=0.0,
         calibrationmethod='sigmoid',
         coef0 = 0.0,
         probability = True,
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
    :param C: trades off misclassification of training examples against simplicity of the decision surface
              low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly 
    :param gamma: parameter defines how far the influence of a single training example reaches
                  low values meaning ‘far’ and high values meaning ‘close’. 
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
               probability = probability,
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
   
    model.fit(train, labels, sample_weight)
      
    predictions = model.predict_proba(test)

    if rescale_pred:
        predictions = rescale_prior(predictions, bincount(labels))
    return predictions  


#This is tree-based feature selection, supposed to discard irrelevant features
#however, the same parameters with feature selection scored lower
def feature_selection(train_data, classes):
    clf = ExtraTreesClassifier()
    train_data_new=clf.fit(train_data, classes).transform(train_data)
    return train_data_new, clf.feature_importances_
    
    
def feature_selection1(train_data, clsses):
    clf=LinearSVC(C=0.01, penalty="l1", dual=False)
    train_data_new = clf.fit(train_data,classes).transform(train_data, classes)
    return train_data_new


def grid_search(train_data, true_classes):
    
    X_2d = train_data[:, :2]
    X_2d = X_2d[true_classes > 0]
    y_2d = true_classes[true_classes > 0]
    y_2d -= 1
    
    #scale the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    X_2d = scaler.fit_transform(X_2d)
    
    # Train classifiers
    #
    # For an initial search, a logarithmic grid with basis
    # 10 is often helpful. Using a basis of 2, a finer
    # tuning can be achieved but at a much higher cost.
    
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(true_classes, n_iter=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(train_data, true_classes)

    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
      
    return grid.best_params_
    
if __name__ == '__main__':

    from utils.loading import get_training_data, get_testing_data
    from validation.crossvalidate import SampleCrossValidator
    import numpy as np
    import pylab as pl

    train_data, true_classes, _ = get_training_data()
    test_data, _ = get_testing_data ()
    
    new_train_data,feat_importance = feature_selection(train_data, true_classes)
    pl.show(feat_importance)
    
    #this takes long to compute, ideally supposed to print the best c and gamma (still computing now)
    #grid_search(train_data, true_classes)

    # no feature selection,it scored lower
    validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)
    
    
    for train, classes, test in validator.yield_cross_validation_sets():
        prediction = svm(train,
                         classes,
                         test)
                         
        validator.add_prediction(prediction)
        
    validator.print_results()
    
    """
    np.save('/home/andruta/anaconda' ,prediction)
    np.savetxt("prediction", prediction, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='')
    """