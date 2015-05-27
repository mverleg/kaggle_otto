
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
from nnet.base_optimize import name_from_file
from nnet.prepare import LogTransform
from nnet.scikit import NNet
from utils.loading import get_training_data, get_testing_data
from settings import LOGS_DIR
from utils.expand_train import expand_from_test
from utils.features import PositiveSparseFeatureGenerator, PositiveSparseRowFeatureGenerator


train, labels = get_training_data()[:2]
test = get_testing_data()[0]
train, labels = expand_from_test(train, labels, test, confidence = 0.9)
gen = PositiveSparseRowFeatureGenerator()
train = gen.fit_transform(train, labels)
test = gen.transform(test, labels)

cpus = max(cpu_count() - 1, 1)
random = RandomState()

opt = RandomizedSearchCV(
	estimator = Pipeline([
		('gen1', PositiveSparseFeatureGenerator(difficult_classes = (2, 3), extra_features = 50)),
		('gen2', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 50)),
		('gen3', PositiveSparseFeatureGenerator(difficult_classes = (1, 9), extra_features = 63)),
		('log', LogTransform()),
		('scale', MinMaxScaler(feature_range = (0, 3))),
		('nn', NNet(
			name = name_from_file(),
			dense1_nonlinearity = 'rectify',
			dense1_init = 'glorot_normal',
			auto_stopping = True,
			max_epochs = 1500,  # binom(n = 4000, p = 0.25)
		)),
	]),
	param_distributions = {
		'nn__batch_size': binom(n = 256, p = 0.5),
		'nn__learning_rate': norm(0.0005, 0.0005),
		'nn__learning_rate_scaling': [1, 10, 100, 1000],
		'nn__momentum': [0, 0.9, 0.99, 0.999],
		'nn__momentum_scaling': [1, 10, 100],
		'nn__dense1_size': randint(low = 100, high = 120),
		'nn__dense2_size': randint(low = 50, high = 90),
		'nn__dense3_size': randint(low = 25, high = 70),
		'nn__dropout0_rate': triang(loc = 0, c = 0, scale = 1),  # beta(a = 0.5, b = 0.5),
		'nn__dropout1_rate': triang(loc = 0, c = 0, scale = 1),
		'nn__dropout2_rate': triang(loc = 0, c = 0, scale = 1),
		'nn__dropout3_rate': triang(loc = 0, c = 0, scale = 1),
		'nn__weight_decay': norm(0.00006, 0.0001),
	},
	fit_params = {
	},
	n_iter = 1,
	n_jobs = cpus,
	scoring = log_loss_scorer,
	iid = False,
	refit = False,
	pre_dispatch = cpus + 2,
	cv = ShuffleSplit(
		n = train.shape[0],
		n_iter = cpus,
		test_size = 0.2,
		random_state = random,
	),
	random_state = random,
)

opt.fit(train, labels)

with open(join(LOGS_DIR, 'random_search_{0:.4f}.json'.format(-opt.best_score_)), 'w+') as fh:
	print 'saving results (no scaling to priors) for top score {0:.4f}:'.format(-opt.best_score_), opt.best_params_
	dump(opt.best_params_, fp = fh, indent = 4)


