
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:50:21 2015

@author: andra
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from math import floor
from numpy import ones, shape, bincount
from utils.preprocess import obtain_class_weights, equalize_class_sizes
from utils.outliers import filter_data
from utils.postprocess import rescale_prior

def AdaBoost(train,
			labels,
			test,
			calibration=0.0,
			calibrationmethod='sigmoid',
			base_estimator= 'DecisionTreeClassifier',
			n_estimators=50,
			learning_rate=1.0,
			algorithm='SAMME',
			random_state=None,
			sample_weight=None,
			outlier_frac=False,
			outlier_method='EE',
			undersample=False,
			rescale_pred= False,
			verbose=0,
			class_weight=None):


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

	if outlier_frac:
		train, labels = filter_data(train, labels, cut_outlier_frac = outlier_frac, method = outlier_method) #remove outliers
	if undersample:
		train, labels = equalize_class_sizes(train, labels)
	if isinstance(sample_weight, str):
		sample_weight = obtain_class_weights(labels, sample_weight)

	N = len(labels)
	trainrows = int((1.0 - calibration) * N)

	model = AdaBoostClassifier(base_estimator = base_estimator,
							   n_estimators=n_estimators,
							   learning_rate=learning_rate,
							   algorithm=algorithm,
							   random_state=random_state)
	if not calibration:
		model.fit(train, labels, sample_weight)
		predictions = model.predict_proba(test)
	elif calibration > 1:
		calibratedmodel = CalibratedClassifierCV(model, calibrationmethod, calibration)
		calibratedmodel.fit(train, labels, sample_weight)
		predictions = calibratedmodel.predict_proba(test)
	else:
		if sample_weight is None:
			sample_weight = ones((len(labels)))
		print 'trainrows', trainrows
		model.fit(train[:trainrows, :], labels[:trainrows],sample_weight[:trainrows])
		calibratedmodel = CalibratedClassifierCV(model, calibrationmethod, "prefit")
		calibratedmodel.fit(train[trainrows:,:], labels[trainrows:], sample_weight = sample_weight[trainrows:])
		predictions = calibratedmodel.predict_proba(test)

	if rescale_pred:
		predictions = rescale_prior(predictions, bincount(labels))
	return predictions

if __name__ == '__main__':

	from utils.loading import get_training_data
	from validation.crossvalidate import SampleCrossValidator

	train_data, true_classes, _ = get_training_data()
	validator = SampleCrossValidator(train_data, true_classes, rounds=1, test_frac=0.1, use_data_frac=1.0)
	for train, classes, test in validator.yield_cross_validation_sets():
		prediction = AdaBoost(train, classes, test, 1, 10, 0.1, verbose = 1)
		#Warning: this will take a while, for faster testing, change the 100 iterations to 1 or something
		validator.add_prediction(prediction)
	validator.print_results()

