
from multiprocessing import cpu_count
from numpy import logspace
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from nnet.oldstyle.base_optimize import name_from_file
from nnet.prepare import LogTransform
from nnet.scikit import NNet
from utils.features import PositiveSparseRowFeatureGenerator, PositiveSparseFeatureGenerator
from utils.loading import get_training_data, get_preproc_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize_parallel import ParallelGridOptimizer


def train_test(train, labels, test, **parameters):
	params = {
		'name': name_from_file(),
		'max_epochs': 600,
		'auto_stopping': True,
		'adaptive_weight_decay': False,
		'save_snapshots_stepsize': None,
		'epoch_steps': None,
		#'dense3_size': 0,
		'momentum_scaling': 1200,
		'dense1_nonlinearity': 'rectify',
		'dense1_init': 'glorot_uniform',
		'dense2_nonlinearity': 'rectify',
		'dense2_init': 'glorot_uniform',
		'batch_size': 128,
		'learning_rate': 0.0003,
		'learning_rate_scaling': 1000,
		'momentum': 0.98,
		'dense1_size': 700,
		'dense2_size': 550,
		'dense3_size': 400,
		#'nn__dense3_size': randint(low = 100, high = 400),
		'dropout0_rate': 0.,
		'dropout1_rate': 0.1,
		'dropout2_rate': 0.45,
		'dropout3_rate': 0.58,
		#'nn__dropout3_rate': triang(loc = 0, c = 0, scale = 1),
		#'nn__weight_decay': norm(0.00006, 0.0001),
	}
	params.update(parameters)
	estimator = Pipeline([
		('row', PositiveSparseRowFeatureGenerator()),
		('gen23', PositiveSparseFeatureGenerator(difficult_classes = (2, 3), extra_features = 40)),
		('gen234', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 40)),
		('gen19', PositiveSparseFeatureGenerator(difficult_classes = (1, 9), extra_features = 63)),
		('log', LogTransform()),  # log should be after integer feats but before dist
		#('distp31', DistanceFeatureGenerator(n_neighbors = 3, distance_p = 1)),
		#('distp52', DistanceFeatureGenerator(n_neighbors = 5, distance_p = 2)),
		('scale03', MinMaxScaler(feature_range = (0, 3))),  # scale should apply to int and float feats
		('nn', NNet(**params)),
	])
	estimator.fit(train, labels)
	prediction = estimator.predict_proba(test)
	return prediction


train_data, true_labels = get_training_data()[:2]
validator = SampleCrossValidator(train_data, true_labels, rounds = 1, test_frac = 0.2, use_data_frac = 1)  # 0.3!!
optimizer = ParallelGridOptimizer(train_test_func = train_test, validator = validator, use_caching = False, process_count = max(cpu_count() - 1, 1), **{
	'weight_decay': logspace(-1, -7, base = 10, num = 30),
}).readygo(save_fig_basename = name_from_file(), log_name = name_from_file() + '_stats.txt')

