"""
Created on Wed jun 10 15:27:20 2015

@author: Tim Janssen
"""
from math import floor
from numpy import bincount
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from utils.postprocess import rescale_prior


def knn(train,
        labels,
        test,
        rescale_pred=False,
        calibration=0.0,
        calibrationmethod='sigmoid',
        n_neighbors=5,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None):

    # Define the parameters of knn model
    model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                 weights=weights,
                                 algorithm=algorithm,
                                 leaf_size=leaf_size,
                                 p=p,
                                 metric=metric,
                                 metric_params=metric_params)

    # if set make a Calibrated model
    if (calibration > 0) and (calibration < 1.0):
        N = len(labels)
        # Determine number of rows used for training
        train_rows = floor((1.0 - calibration) * N)

        # Make a calibrated model
        model.fit(train[:train_rows, :], labels[:train_rows])
        model = CalibratedClassifierCV(model, calibrationmethod, "prefit")

        train = train[train_rows:, :]
        labels = labels[train_rows:]

    elif calibration > 1:
        model = CalibratedClassifierCV(model, calibrationmethod, calibration)

    # Fit the model to the data
    model.fit(train, labels)

    # Make predictions of the classes
    predictions = model.predict_proba(test)

    # if set Rescale predictions to prior
    if rescale_pred:
        predictions = rescale_prior(predictions, bincount(labels))

    return predictions

if __name__ == '__main__':

    from utils.loading import get_training_data
    from validation.crossvalidate import SampleCrossValidator

    train_data, true_classes, _ = get_training_data()
    validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)
    for train, classes, test in validator.yield_cross_validation_sets():
        prediction = knn(train,
                         classes,
                         test,
                         rescale_pred=False,
                         calibration=0.0,
                         calibrationmethod='sigmoid',
                         n_neighbors=5,
                         weights='uniform',
                         algorithm='auto',
                         p=2,
                         metric='minkowski',
                         metric_params=None)
        validator.add_prediction(prediction)
    validator.print_results()