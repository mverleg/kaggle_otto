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
    return model.predict_proba(test)


if __name__ == '__main__':

    from utils.loading import get_training_data
    from validation.crossvalidate import SampleCrossValidator

    train_data, true_classes, features = get_training_data()
    validator = SampleCrossValidator(train_data, true_classes, rounds=2, test_frac=0.3, use_data_frac=0.7)
    for train, classes, test in validator.yield_cross_validation_sets():
        prediction = gradientBoosting(train, classes, test, 5, verbose = 0)
        validator.add_prediction(prediction)
    validator.print_results()

