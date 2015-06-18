
from multiprocessing import cpu_count
from numpy import load, log10, float64
from os.path import join
from sklearn.calibration import CalibratedClassifierCV
from nndist.distance import DistanceClassifier
from settings import BASE_DIR
from utils.postprocess import scale_to_priors
from validation.crossvalidate import SampleCrossValidator
from validation.optimize_parallel import ParallelGridOptimizer


true_labels = load(join(BASE_DIR, 'data', 'trainclas.npy'))
train_data = load(join(BASE_DIR, 'data', 'trainmat.npy'))[:, 1:].astype(float64)
validation_data = load(join(BASE_DIR, 'data', 'testmat.npy'))[:, 1:].astype(float64)


def train_test(train, labels, test, n_neighbors, distance_p, use_log = False, use_autoscale = False, use_calibration = False):
	if use_log:
		train, test = log10(1 + train), log10(1 + test)
	if use_autoscale:
		train /= train.max(0)
		test /= test.max(0)
	clf = DistanceClassifier(n_neighbors = n_neighbors, distance_p = distance_p)
	if use_calibration:
		clf = CalibratedClassifierCV(clf, cv = 3)
	clf.fit(train, labels)
	probs = clf.predict_proba(test)
	probs = scale_to_priors(probs)
	return probs

#probs = train_test(train_data[::10, :], true_labels[::10], validation_data[::10], n_neighbors = 1, distance_p = 1, use_log = False)
#print probs.shape

validator = SampleCrossValidator(train_data, true_labels, rounds = 5, test_frac = 0.1, use_data_frac = 0.4)
optimizer = ParallelGridOptimizer(train_test_func = train_test, validator = validator, use_caching = False, process_count = max(cpu_count() - 1, 1),
    n_neighbors = [4, 8, 12, 20, 32],
	distance_p = 2,
	use_log = True,
    use_autoscale = False,
    use_calibration = [True, False],
).readygo()


"""
pos     loss      n neighbors       use autoscale     use log           distance p
 1    0.7255       20                False             True              2
 2    0.7260       8                 False             True              2
 3    0.7265       12                False             True              2
 4    0.7279       20                True              True              2
 5    0.7291       4                 False             True              2
 6    0.7300       8                 True              True              2
 7    0.7301       12                True              True              2
 8    0.7318       32                False             True              2
 9    0.7331       4                 True              True              2
10    0.7334       32                True              True              2
11    0.7359       12                False             False             1
12    0.7363       20                False             False             1
13    0.7368       8                 False             False             1
"""