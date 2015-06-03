
from json import dump
from multiprocessing import cpu_count
from os.path import join
from scipy.stats import binom, norm, triang, randint
from numpy.random import RandomState
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics.scorer import log_loss_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from nnet.oldstyle.base_optimize import name_from_file
from nnet.prepare import LogTransform
from nnet.scikit import NNet
from settings import LOGS_DIR, VERBOSITY
from utils.features import PositiveSparseFeatureGenerator, PositiveSparseRowFeatureGenerator, DistanceFeatureGenerator
from utils.loading import get_preproc_data, get_training_data, get_testing_data


#train, labels, test = get_preproc_data(Pipeline([
#	('row', PositiveSparseRowFeatureGenerator()),
#	('distp31', DistanceFeatureGenerator(n_neighbors = 3, distance_p = 1)),
#	('distp52', DistanceFeatureGenerator(n_neighbors = 5, distance_p = 2)),
#]), expand_confidence = 0.9)
train, labels = get_training_data()[:2]
test = get_testing_data()[0]

#cpus = max(cpu_count() - 1, 1)
#random = RandomState()

opt = RandomizedSearchCV(
	estimator = Pipeline([
		#('gen23', PositiveSparseFeatureGenerator(difficult_classes = (2, 3), extra_features = 40)),
		#('gen234', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 40)),
		#('gen19', PositiveSparseFeatureGenerator(difficult_classes = (1, 9), extra_features = 63)),
		('log', LogTransform()),
		('scale03', MinMaxScaler(feature_range = (0, 3))),
		('nn', NNet(**{
			'dense1_nonlinearity': 'rectify',
			'dense1_init': 'glorot_normal',
			'max_epochs': 500,
			'learning_rate': 0.000676,
			'learning_rate_scaling': 30,
			'momentum': 0.9,
			'momentum_scaling': 10,
			'dense1_size': 350,
			'dense2_size': 463,
			'dense3_size': 329,
			'dropout0_rate': 0.06,
			'dropout1_rate': 0.56,
			'dropout2_rate': 0.19,
			'dropout3_rate': 0.04,
			'adaptive_weight_decay': False,
			'auto_stopping': True, ##
			'save_snapshots_stepsize': None,
			'epoch_steps': None,
		})),
	]),
	param_distributions = {
		'nn__name': ['nn{0:03d}'.format(k) for k in range(10000)],
		#'nn__batch_size': binom(n = 256, p = 0.5),
		#'nn__learning_rate': norm(0.0005, 0.0005),
		#'nn__learning_rate_scaling': [1, 10, 100, 1000],
		#'nn__momentum': [0, 0.9, 0.99, 0.999],
		#'nn__momentum_scaling': [1, 10, 100],
		#'nn__dense1_size': randint(low = 100, high = 800),
		#'nn__dense2_size': randint(low = 50, high = 650),
		#'nn__dense3_size': randint(low = 25, high = 500),
		#'nn__dropout0_rate': triang(loc = 0, c = 0, scale = 1),
		#'nn__dropout1_rate': triang(loc = 0, c = 0, scale = 1),
		#'nn__dropout2_rate': triang(loc = 0, c = 0, scale = 1),
		#'nn__dropout3_rate': triang(loc = 0, c = 0, scale = 1),
		#'nn__weight_decay': norm(0.00006, 0.0001),
		'nn__weight_decay': [1.0e-05, 0],
	},
	fit_params = {
	},
	n_iter = 10,
	n_jobs = 10,
	scoring = log_loss_scorer,
	iid = False,
	refit = False,
	#pre_dispatch = cpus + 2,
	cv = ShuffleSplit(
		n = train.shape[0],
		n_iter = 1,
		test_size = 0.2,
		#random_state = random,
	),
	#random_state = random,
	verbose = bool(VERBOSITY),
	error_score = -1000000,
)

opt.fit(train, labels)

with open(join(LOGS_DIR, 'debug_{0:.4f}.json'.format(-opt.best_score_)), 'w+') as fh:
	print 'saving results (no scaling to priors) for top score {0:.4f}:'.format(-opt.best_score_), opt.best_params_
	dump(opt.best_params_, fp = fh, indent = 4)



