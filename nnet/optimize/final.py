
from json import dump
from multiprocessing import cpu_count
from os.path import join
from scipy.stats import norm, uniform
from numpy.random import RandomState
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from nnet.oldstyle.base_optimize import name_from_file
from nnet.prepare import LogTransform
from nnet.score_logging import get_logloss_loggingscorer
from settings import OPTIMIZE_RESULTS_DIR
from nnet.scikit import NNet
from settings import LOGS_DIR, VERBOSITY, SUBMISSIONS_DIR, PRIORS
from utils.features import PositiveSparseFeatureGenerator, PositiveSparseRowFeatureGenerator
from utils.ioutil import makeSubmission
from utils.loading import get_preproc_data
from utils.postprocess import scale_to_priors


train, labels, test = get_preproc_data(None, expand_confidence = None)

cpus = max(cpu_count() - 1, 1)
random = RandomState()

opt = RandomizedSearchCV(
	estimator = Pipeline([
		('row', PositiveSparseRowFeatureGenerator()),
		('gen23', PositiveSparseFeatureGenerator(difficult_classes = (2, 3), extra_features = 40)),
		('gen234', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 40)),
		('gen19', PositiveSparseFeatureGenerator(difficult_classes = (1, 9), extra_features = 63)),
		('log', LogTransform()), # log should be after integer feats but before dist
		#('distp31', DistanceFeatureGenerator(n_neighbors = 3, distance_p = 1)),
		#('distp52', DistanceFeatureGenerator(n_neighbors = 5, distance_p = 2)),
		('scale03', MinMaxScaler(feature_range = (0, 3))), # scale should apply to int and float feats
		('nn', NNet(**{
			'max_epochs': 700,
			'auto_stopping': True,
			'adaptive_weight_decay': False,
			'save_snapshots_stepsize': None,
			'epoch_steps': None,
			'dense1_size': 400,
			'dense2_size': 300,
			'dense3_size': 300,
			'dense1_nonlinearity': 'rectify',
			'dense1_init': 'glorot_uniform',  # uniform to reduce randomness
			'momentum': 0.98,
			'momentum_scaling': 10,
			'learning_rate_scaling': 300,
			'batch_size': 128,
			'dropout0_rate': 0,
		})),
	]),
	param_distributions = {
		'nn__name': ['final_{0:s}_{1:03d}'.format(name_from_file(), k) for k in range(10000)],
		'nn__learning_rate': norm(0.0003, 0.0001),
		'nn__dropout1_rate': uniform(loc = 0.00, scale = 0.25),
		'nn__dropout2_rate': uniform(loc = 0.30, scale = 0.60-0.30),
		'nn__dropout3_rate': uniform(loc = 0.40, scale = 0.80-0.40),  #todo
	},
	fit_params = {
		'nn__random_sleep': 80,
	},
	n_iter = cpus,
	n_jobs = cpus,
	scoring = get_logloss_loggingscorer(
		filename = join(OPTIMIZE_RESULTS_DIR, '{0:s}2.log'.format(name_from_file())),
		treshold = None,
	),
	iid = False,
	refit = False,
	pre_dispatch = cpus + 2,
	cv = ShuffleSplit(
		n = train.shape[0],
		n_iter = 1,
		test_size = 0.2,
		random_state = random,
	),
	random_state = random,
	verbose = bool(VERBOSITY),
	error_score = 'raise',
)

opt.fit(train, labels)

print '> saving results for top score {0:.4f} (adding rescaling to priors):'.format(-opt.best_score_), opt.best_params_
try:
	with open(join(LOGS_DIR, 'random_search_{0:.4f}.json'.format(-opt.best_score_)), 'w+') as fh:
		dump(opt.best_params_, fp = fh, indent = 4)
	probs = opt.best_estimator_.predict_proba(test)
	probs = scale_to_priors(probs, priors = PRIORS)
	makeSubmission(probs, fname = join(SUBMISSIONS_DIR, 'random_search_{0:.4f}.csv'.format(-opt.best_score_)), digits = 8)
except Exception as err:
	print 'Something went wrong while storing results. Maybe refit isn\'t enabled?'
	print err


