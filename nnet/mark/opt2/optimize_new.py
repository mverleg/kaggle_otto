
from os.path import basename, splitext
from sys import modules
from nnet.make_net import make_net
from nnet.prepare import normalize_data
from utils.loading import get_training_data
from utils.outliers import filter_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize_parallel import ParallelGridOptimizer


def train_test_NN(train, classes, test, **parameters):
	outlier_frac = parameters.pop('outlier_frac', 0)
	outlier_method = parameters.pop('outlier_method', 'OCSVM')
	normalize_log = parameters.pop('normalize_log', True)
	if outlier_frac:
		train, classes = filter_data(train, classes, cut_outlier_frac = outlier_frac, method = outlier_method)  # remove ourliers
	train, norms = normalize_data(train, use_log = normalize_log)  # also converts to floats
	test = normalize_data(test, norms = norms, use_log = normalize_log)[0]  # scale test data in the same way as train data
	net = make_net(**parameters)
	net.fit(train, classes - 1)
	return net.predict_proba(test)


name = '{0:s}.log'.format(splitext(basename(getattr(modules['__main__'], '__file__', 'optimize.default')))[0])  # automatic based on filename

train_data, true_classes, features = get_training_data()  # load the train data
validator = SampleCrossValidator(train_data, true_classes, rounds = 1, test_frac = 0.2, use_data_frac = 1)
optimizer = ParallelGridOptimizer(train_test_func = train_test_NN, validator = validator, use_caching = False,
	name = name,                      # just choose something sensible
	dense1_nonlinearity = 'tanh',     # ['tanh', 'sigmoid', 'rectify', 'leaky2', 'leaky20' 'softmax']
	dense1_init = 'glorot_uniform',   # ['orthogonal', 'sparse', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
	dense1_size = 128,                # [30, 25, 80, 120, 180]
	dense2_size = None,
	dense3_size = None,
	learning_rate = 0.001,            # initial learning reate
	learning_rate_scaling = 10,       # pogression over time; 0.1 scaled by 10 is 0.01
	momentum = 0.99,                  # initial momentum
	momentum_scaling = 10,            # 0.9 scaled by 10 is 0.99
	dropout1_rate = 0.5,              # [0, 0.5]
	dropout2_rate = 0.5,
	weight_decay = 0,                 # constrain the weights to avoid overfitting
	max_epochs = 2000,                # it terminates when overfitting or increasing, so just leave high
	output_nonlinearity = 'softmax',  # just keep softmax
	auto_stopping = True,             # stop training automatically if it seems to be failing
	outlier_method = 'OCSVM',         # method for outlier removal ['OCSVM', 'EE']
	outlier_frac = None,              # which fraction of each class to remove as outliers
	normalize_log = True,             # use logarithm for normalization
).readygo(topprint = 20, save_fig_basename = name, log_name = name, only_show_top = True)


