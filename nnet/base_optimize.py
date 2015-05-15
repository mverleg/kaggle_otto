
from copy import copy
from nnet.train_test import train_test_NN, make_pretrain
from os.path import basename, splitext, join
from sys import modules
from settings import PRETRAIN_DIR
from utils.expand_train import expand_from_test
from utils.loading import get_training_data, get_testing_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer, is_nonstr_iterable
from validation.optimize_parallel import ParallelGridOptimizer


def name_from_file(pth = getattr(modules['__main__'], '__file__', 'optimize.default')):
	"""
		Get the base nname from a file path (the main file by default).
	"""
	return '{0:s}'.format(splitext(basename(pth))[0])


DEFAULT_PARAMS = {
	'name': name_from_file(),
	'dense1_nonlinearity': 'rectify',   # tanh, sigmoid, rectify, leaky2, leaky20, softmax
	'dense1_init': 'glorot_normal',     # orthogonal, sparse, glorot_normal, glorot_uniform, he_normal, he_uniform
	'dense1_size': 128,                 # hidden neurons in layer (30~1000)
	'dense2_size': None,
	'dense3_size': None,
	'learning_rate': 0.001,             # initial learning rate (learning rate is effectively higher for higher momentum)
	'learning_rate_scaling': 1000,      # progression over time; 0.1 scaled by 10 is 0.01
	'momentum': 0.9,                    # initial momentum
	'momentum_scaling': 100,            # 0.9 scaled by 10 is 0.99
	'dropout1_rate': 0,                 # [0, 0.5]
	'dropout2_rate': None,              # inherit dropout1_rate if dense3 exists
	'weight_decay': 0,                  # constrain the weights to avoid overfitting
	'max_epochs': 1000,                 # it terminates when overfitting or increasing, so just leave high
	'auto_stopping': True,              # stop training automatically if it seems to be failing
	'pretrain': None,                   # use pretraining? (True/False / filename / None[= when possible])
	'outlier_method': 'EE',             # method for outlier removal ['OCSVM', 'EE']
	'outlier_frac': None,               # which fraction of each class to remove as outliers
	'normalize_log': True,              # use logarithm for normalization
	'use_calibration': False,           # use calibration of probabilities
	'use_rescale_priors': True,         # rescale predictions to match priors
	'extra_feature_count': 0,           # how many new features to generate
	'extra_feature_seed': 0,            # a seed for the feature generation
	'save_snapshots_stepsize': None,    # save snapshot of the network every X epochs
	'test_data_confidence': None,       # how confident test samples should be to be added to train ([0.5 - 1.0] or None)
}


def optimize_NN(name = name_from_file(), rounds = 1, debug = False, use_caching = True, train_test_func = train_test_NN, **special_params):
	"""
		Some default code for optimization, adding default parameters and debug, and using mostly other classes to do the rest of the work.
	"""
	"""
		Default parameters.
	"""
	for key in special_params.keys():
		assert key in DEFAULT_PARAMS.keys(), '"{0:s}" is not a known parameter'.format(key)
	params = copy(DEFAULT_PARAMS)
	params.update(special_params)

	"""
		Load data.
	"""
	train_data, true_labels, features = get_training_data()

	"""
		Pre-training.
	"""
	if params['pretrain'] or params['pretrain'] is None:
		layer_sizes = [params['extra_feature_count'] or 0, params['dense1_size'] or 0, params['dense2_size'] or 0, params['dense2_size'] or 0, params['dropout1_rate'] or 0, params['dropout2_rate'] or 0]
		if any(is_nonstr_iterable(nr) for nr in layer_sizes):
			""" Different layouts, so no pre-training. """
			if params['pretrain'] is None:
				print 'No pre-training since layer sizes are not constant.'
				params['pretrain'] = False
			else:
				raise AssertionError('Pre-training is not available when there are different network layouts (e.g. different numbers of neurons or features).')
		else:
			""" Constant layout, so can use pre-training. """
			if params['pretrain'] is None or params['pretrain'] is True:
				params['pretrain'] = join(PRETRAIN_DIR, 'pt{0:s}.net.npz'.format('x'.join(str(nr) for nr in layer_sizes if nr is not None)))
			make_pretrain(params['pretrain'], train_data, true_labels, **params)

	"""
		The actual optimization, optionally in debug mode (non-parallel for stacktrace and resource use).
	"""
	validator = SampleCrossValidator(train_data, true_labels, rounds = rounds, test_frac = 0.2, use_data_frac = 1)
	if debug:
		optimizer = GridOptimizer(validator, use_caching = use_caching, **params)
		for subparams, train, labels, test in optimizer.yield_batches():
			optimizer.register_results(train_test_func(train, labels, test, **subparams))
		optimizer.print_plot_results()
	else:
		ParallelGridOptimizer(train_test_func = train_test_func, validator = validator, use_caching = use_caching, **params
			).readygo(save_fig_basename = name, log_name = name + '.log', only_show_top = True)


