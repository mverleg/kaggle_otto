
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


def train_test(train, labels, test, n_neighbors, distance_p, use_log = False):
	if use_log:
		train, test = log10(1 + train), log10(1 + test)
	dclf = DistanceClassifier(n_neighbors = n_neighbors, distance_p = distance_p)
	cclf = CalibratedClassifierCV(dclf, cv = 3)
	cclf.fit(train, labels)
	probs = cclf.predict_proba(test)
	print 'rescaling to priors'
	probs = scale_to_priors(probs)
	return probs

#probs = train_test(train_data[::10, :], true_labels[::10], validation_data[::10], n_neighbors = 1, distance_p = 1, use_log = False)
#print probs.shape

validator = SampleCrossValidator(train_data, true_labels, rounds = 3, test_frac = 0.2, use_data_frac = 0.1)
optimizer = ParallelGridOptimizer(train_test_func = train_test, validator = validator, use_caching = False, process_count = max(cpu_count() - 1, 1),
	n_neighbors = [1, 2, 4, 6, 8, 10, 11],
	distance_p = [1, 2],
	use_log = [True, False],
).readygo()


