from os.path import basename, splitext
from sys import modules

from nnet.oldstyle import make_net
from nnet.prepare import normalize_data, equalize_class_sizes
from utils.loading import get_training_data
from utils.outliers import filter_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize_parallel import ParallelGridOptimizer


def train_test_NN(train, classes, test, **parameters):

	train, classes = equalize_class_sizes(train, classes)
	train, classes = filter_data(train, classes, cut_outlier_frac = 0.06, method = 'OCSVM')  # remove ourliers
	train = normalize_data(train, use_log = True)[0]  # also converts to floats
	test = normalize_data(test, use_log = True)[0]

	parameters['dense2_nonlinearity'] = parameters['dense1_nonlinearity']  # hack1
	parameters['dense2_init'] = parameters['dense1_init']  # hack2
	net = make_net(**parameters)
	net.fit(train, classes - 1)
	return net.predict_proba(test)



# one big job for size1 + size2 + dropout : (180, 180, 0)
# one job for nonlinearity1 + initialization1 (he_uniform, tanh)
# one job for learning rate + learning rate scaling + momentum + momentum scaling (split if too many) (0.001,10  / 0.99,10)

name = '{0:s}.log'.format(splitext(basename(getattr(modules['__main__'], '__file__', 'optimize.default')))[0])  # automatic based on filename
print("Name: " + name)

name = name                      # just choose something sensible
dense1_size = 180                # [30, 25, 80, 120, 180]  - default 60
dense1_nonlinearity = ['tanh', 'sigmoid', 'rectify', 'leaky2', 'leaky20', 'softmax']
                     #'leaky20'  # ['tanh', 'sigmoid', 'rectify', 'leaky2', 'leaky20', 'softmax']
dense1_init =  ['orthogonal', 'sparse', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
             #'orthogonal'       # ['orthogonal', 'sparse', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
dense2_size = 180                # [30, 25, 80, 120, 180]
dense2_nonlinearity = 'leaky20'  # this is coupled to dense1_nonlinearity through hack#1
dense2_init = 'orthogonal'       # idem hack2
learning_rate = 0.001            # [0.1, 0.01, 0.001, 0.0001]
learning_rate_scaling = 10  # 10 (used to be 100)     # [1, 10, 100]
momentum = 0.99                   # [0, 0.9, 0.99]
momentum_scaling = 10           # [1, 10, 100]
dropout1_rate = 0                # [0, 0.5]
dropout2_rate = None
weight_decay = 0                 # doesn't work
max_epochs = 3000                # it terminates when overfitting or increasing, so just leave high
output_nonlinearity = 'softmax'  # just keep softmax
verbosity = 1                    # farts output hard, but at least you get some idea of progress
auto_stopping = True

baseparams = { "name" : name, "verbosity" : verbosity} #, 'dense1_nonlinearity' : dense1_nonlinearity, 'dense1_init': dense1_init}
testparams = baseparams.copy()
testparams.update(  {"dense1_size" : dense1_size, "dense2_size" : dense2_size, "dropout1_rate" : dropout1_rate} )
testparams.update(  {'dense1_init' : dense1_init, 'dense1_nonlinearity' : dense1_nonlinearity } )
testparams.update ( {'momentum' : momentum, 'momentum_scaling' : momentum_scaling} )
testparams.update ( {'learning_rate' : learning_rate, 'learning_rate_scaling' : learning_rate_scaling} )

train_data, true_classes, features = get_training_data()  # load the train data

from numpy import shape
print shape(train_data)

validator = SampleCrossValidator(train_data, true_classes, rounds = 1, test_frac = 0.1, use_data_frac = 1)
optimizer = ParallelGridOptimizer(train_test_func = train_test_NN, validator = validator, use_caching = False, process_count = 36, **testparams)
print("Declared optimizer complete")
optimizer.readygo(topprint = 100, save_fig_basename = name, log_name = name, only_show_top = True)



#train_test_NN(train_data, true_classes, train_data, **baseparams)


