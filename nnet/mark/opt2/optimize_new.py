
from os.path import basename, splitext
from sys import modules
from nnet.make_net import make_net
from nnet.prepare import normalize_data, equalize_class_sizes
from utils.loading import get_training_data
from utils.outliers import filter_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize_parallel import ParallelGridOptimizer



def train_test_NN(train, classes, test, **parameters):
	parameters['dense2_nonlinearity'] = parameters['dense1_nonlinearity']  # hack1
	parameters['dense2_init'] = parameters['dense1_init']  # hack2
	outlier_frac = parameters.pop('outlier_frac')
	normalize_log = parameters.pop('normalize_log')
	train, classes = filter_data(train, classes, cut_outlier_frac = outlier_frac, method = 'OCSVM')  # remove ourliers
	train, norms = normalize_data(train, use_log = normalize_log)  # also converts to floats
	test = normalize_data(test, norms = norms, use_log = normalize_log)[0]  # scale test data in the same way as train data
	net = make_net(**parameters)
	net.fit(train, classes - 1)
	return net.predict_proba(test)


name = '{0:s}.log'.format(splitext(basename(getattr(modules['__main__'], '__file__', 'optimize.default')))[0])  # automatic based on filename

train_data, true_classes, features = get_training_data()  # load the train data
validator = SampleCrossValidator(train_data, true_classes, rounds = 4, test_frac = 0.2, use_data_frac = 1)
optimizer = ParallelGridOptimizer(train_test_func = train_test_NN, validator = validator, use_caching = True,
	name = name,                      # just choose something sensible
	dense1_size = 180,                # [30, 25, 80, 120, 180]
	dense1_nonlinearity = 'tanh',     # ['tanh', 'sigmoid', 'rectify', 'leaky2', 'leaky20' 'softmax']
	dense1_init = 'glorot_uniform',   # ['orthogonal', 'sparse', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
	dense2_size = 180,                # [30, 25, 80, 120, 180]
	dense2_nonlinearity = 'tanh',     # this is coupled to dense1_nonlinearity through hack#1
	dense2_init = 'glorot_uniform',   # idem hack2
	learning_rate = 0.001,            # initial learning reate
	learning_rate_scaling = 10,       # pogression over time; 0.1 scaled by 10 is 0.01
	momentum = 0.99,                  # initial momentum
	momentum_scaling = 10,            # 0.9 scaled by 10 is 0.99
	dropout1_rate = 0.5,              # [0, 0.5]
	dropout2_rate = None,
	weight_decay = 0,                 # doesn't work
	max_epochs = 50,                  # it terminates when overfitting or increasing, so just leave high
	output_nonlinearity = 'softmax',  # just keep softmax
	auto_stopping = True,             # stop training automatically if it seems to be failing
	outlier_frac = 0.06,              # which fraction of each class to remove as outliers
	normalize_log = True,             # use logarithm for normalization
).readygo(topprint = 20, save_fig_basename = name, log_name = name, only_show_top = True)


