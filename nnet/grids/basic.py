
from json import dump
from multiprocessing import cpu_count
from os.path import join
from nolearn.lasagne import NeuralNet, BatchIterator
from scipy.stats import norm, uniform
from numpy import float32, logspace
from numpy.random import RandomState
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from theano import shared
from nnet.oldstyle.base_optimize import name_from_file
from nnet.prepare import LogTransform
from nnet.score_logging import get_logloss_loggingscorer
from nnet.weight_decay import WeightDecayObjective
from settings import OPTIMIZE_RESULTS_DIR, NCLASSES
from nnet.scikit import NNet
from settings import LOGS_DIR, VERBOSITY, SUBMISSIONS_DIR, PRIORS
from utils.features import PositiveSparseFeatureGenerator, PositiveSparseRowFeatureGenerator
from utils.ioutil import makeSubmission
from utils.loading import get_preproc_data, get_training_data
from utils.postprocess import scale_to_priors
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.objectives import Objective
from lasagne.updates import nesterov_momentum
from lasagne.init import HeUniform, Constant
from lasagne.nonlinearities import LeakyRectify, softmax
from functools import partial
from validation.crossvalidate import SampleCrossValidator
from validation.optimize_parallel import ParallelGridOptimizer


train_data, true_labels = get_training_data()[:2]
true_labels -= 1

cpus = max(cpu_count() - 1, 1)
random = RandomState()


def train_test(train, labels, test, weight_decay):
	net = NeuralNet(
		layers = [
			('input', InputLayer),
			('dropout0', DropoutLayer),
			('dense1', DenseLayer),
			('dropout1', DropoutLayer),
			('dense2', DenseLayer),
			('dropout2', DropoutLayer),
			('dense3', DenseLayer),
			('dropout3', DropoutLayer),
			('output', DenseLayer),
		],
		update = nesterov_momentum,
		loss = None,
		objective = partial(WeightDecayObjective, weight_decay = weight_decay),
		regression = False,
		max_epochs = 600,
		eval_size = 0.1,
		#on_epoch_finished = None,
		#on_training_finished = None,
		verbose = bool(VERBOSITY),
		input_shape = (None, train.shape[1]),
		output_num_units = NCLASSES,
		dense1_num_units = 700,
		dense2_num_units = 1000,
		dense3_num_units = 700,
		dense1_nonlinearity = LeakyRectify(leakiness = 0.1),
		dense2_nonlinearity = LeakyRectify(leakiness = 0.1),
		dense3_nonlinearity = LeakyRectify(leakiness = 0.1),
		output_nonlinearity = softmax,
		dense1_W = HeUniform(),
		dense2_W = HeUniform(),
		dense3_W = HeUniform(),
		dense1_b = Constant(0.),
		dense2_b = Constant(0.),
		dense3_b = Constant(0.),
		output_b = Constant(0.),
		dropout0_p = 0.1,
		dropout1_p = 0.6,
		dropout2_p = 0.6,
		dropout3_p = 0.6,
		update_learning_rate = shared(float32(0.02)), #
		update_momentum = shared(float32(0.9)), #
		batch_iterator_train = BatchIterator(batch_size = 128),
		batch_iterator_test = BatchIterator(batch_size = 128),
	)
	net.fit(train, labels)
	return net.predict_proba(test)


validator = SampleCrossValidator(train_data, true_labels, rounds = 1, test_frac = 0.2, use_data_frac = 1)  # 0.3!!
optimizer = ParallelGridOptimizer(train_test_func = train_test, validator = validator, use_caching = False, process_count = max(cpu_count() - 1, 1), **{
	'weight_decay': logspace(-1, -7, base = 10, num = 30),
}).readygo(save_fig_basename = name_from_file(), log_name = name_from_file() + '_stats.txt')


