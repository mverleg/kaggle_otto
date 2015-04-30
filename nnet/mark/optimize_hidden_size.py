
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
	net = make_net(**parameters)
	net.fit(train, classes - 1)
	return net.predict_proba(test)


name = '{0:s}.log'.format(splitext(basename(getattr(modules['__main__'], '__file__', 'optimize.default')))[0])  # automatic based on filename

train_data, true_classes, features = get_training_data()  # load the train data
train_data, true_classes = equalize_class_sizes(train_data, true_classes)
train_data, true_classes = filter_data(train_data, true_classes, cut_outlier_frac = 0.06, method = 'OCSVM')  # remove ourliers
train_data = normalize_data(train_data, use_log = True)[0]  # also converts to floats
validator = SampleCrossValidator(train_data, true_classes, rounds = 100, test_frac = 0.2, use_data_frac = 1)
optimizer = ParallelGridOptimizer(train_test_NN, validator = validator, use_caching = True,
	name = name,            # just choose something sensible
	dense1_size = [30, 25, 80, 120],  # [30, 25, 80, 120, 200]
	dense1_nonlinearity = 'leaky20',  # ['tanh', 'sigmoid', 'rectify', 'leaky2', 'leaky20' 'softmax']
	dense1_init = 'orthogonal',       # ['orthogonal', 'sparse', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
	dense2_size = [30, 25, 80, 120],  # [30, 25, 80, 120, 200]
	dense2_nonlinearity = 'leaky20',  # this is coupled to dense1_nonlinearity through hack#1
	dense2_init = 'orthogonal',       # idem hack2
	learning_rate = 0.001,            # [0.1, 0.01, 0.001, 0.0001]
	learning_rate_scaling = 100,      # [1, 10, 100, 1000]
	momentum = 0.9,                   # [0, 0.9, 0.99]
	momentum_scaling = 100,           # [1, 10, 100, 1000]
	dropout1_rate = 0.5,              # [0, 0.5]
	dropout2_rate = None,
	weight_decay = 0,                 # doesn't work
	max_epochs = 3000,                # it terminates when overfitting or increasing, so just leave high
	output_nonlinearity = 'softmax',  # just keep softmax
).readygo(topprint = 100, save_fig_basename = name, log_name = name, only_show_top = True)


