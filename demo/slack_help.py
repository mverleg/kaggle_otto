
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from nnet.hue.split_data import split_data
from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize_parallel import ParallelGridOptimizer


# run this script with -v after it to see more information


def calibrated_random_forest(trainFeatures, classLabels, testFeatures, criterion = 'gini', n_estimators = 200, max_features = 'auto', n_jobs = 1):
	trainSet1, labelSet1, trainSet2, labelSet2 = split_data(trainFeatures, classLabels, 0.3)
	rf = RandomForestClassifier(n_estimators=n_estimators, criterion = criterion, max_features = max_features, n_jobs = n_jobs)
	clf = rf.fit(trainSet1, labelSet1)
	calibrated_clf_sigmoid = CalibratedClassifierCV(clf, method = 'sigmoid', cv = 'prefit')
	calibrated_clf_sigmoid.fit(trainSet2, labelSet2)
	prob = calibrated_clf_sigmoid.predict_proba(testFeatures)
	return prob


train_data, true_labels = get_training_data()[:2]
validator = SampleCrossValidator(train_data, true_labels, rounds = 1, test_frac = 0.1, use_data_frac = 1)
optimizer = ParallelGridOptimizer(calibrated_random_forest, validator = validator,  use_caching = True,
	criterion = 'gini',
	n_estimators = [100, 150, 200],  # change too [100, 150, 200] to compare those values
	max_features = 'auto',
	n_jobs = 1
)
optimizer.readygo()


