"""
Created on Thu Apr 23 20:44:30 2015

@author: Fenno, Tim
"""

import sklearn.ensemble as ske

def randomForest(train, labels, test, n_estimators = 200, max_depth = 35, n_jobs=-1, verbose = 0):
    """
    Trains a model by giving it a feature matrix, as well as the labels (the ground truth)
    then using that model, predicts the given test samples
    output is 9 probabilities, one for each class
    """
    # Todo: fix for class imbalance (class_weight="auto")
    model = ske.RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth, n_jobs= n_jobs, verbose = verbose).fit(train, labels)
    return model.predict_proba(test)


if __name__ == '__main__':

    from utils.loading import get_training_data
    from utils.outliers import filter_data
    from validation.crossvalidate import SampleCrossValidator

    train_data, true_classes, features = get_training_data()
    filtered_data, filtered_classes = filter_data(train_data, true_classes, cut_outlier_frac = 0.06, method = 'EE')
    validator = SampleCrossValidator(filtered_data, filtered_classes, rounds = 2, test_frac = 0.2, use_data_frac = 1)
    for train, classes, test in validator.yield_cross_validation_sets():
        prediction = randomForest(train, classes, test, 200, 35, verbose = 0)
        validator.add_prediction(prediction)
    validator.print_results()