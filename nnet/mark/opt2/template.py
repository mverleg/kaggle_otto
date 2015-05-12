
from nnet.train_test import train_test_NN, make_pretrain
from os.path import basename, splitext, join
from sys import modules
from settings import PRETRAIN_DIR
from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize_parallel import ParallelGridOptimizer


train_data, true_labels, features = get_training_data()
name = '{0:s}.log'.format(splitext(basename(getattr(modules['__main__'], '__file__', 'optimize.default')))[0])
pretrain = join(PRETRAIN_DIR, 'pt_128.net.npz')
params = {
	'name': name,
	'dense1_nonlinearity': 'tanh',     # tanh, sigmoid, rectify, leaky2, leaky20, softmax
	'dense1_init': 'glorot_uniform',   # orthogonal, sparse, glorot_normal, glorot_uniform, he_normal, he_uniform
	'dense1_size': 128,                # [30, 25, 80, 120, 180]
	'dense2_size': None,
	'dense3_size': None,
	'learning_rate': 0.0001,           # initial learning reate
	'learning_rate_scaling': 10,       # pogression over time; 0.1 scaled by 10 is 0.01
	'momentum': 0.99,                  # initial momentum
	'momentum_scaling': 50,            # 0.9 scaled by 10 is 0.99
	'dropout1_rate': 0,                # [0, 0.5]
	'dropout2_rate': 0,
	'weight_decay': 0,                 # constrain the weights to avoid overfitting
	'max_epochs': 1000,                # it terminates when overfitting or increasing, so just leave high
	'output_nonlinearity': 'softmax',  # just keep softmax
	'auto_stopping': True,             # stop training automatically if it seems to be failing
	'pretrain': pretrain,              # use pretraining? (True for automatic, filename for specific)
	'outlier_method': 'OCSVM',         # method for outlier removal ['OCSVM', 'EE']
	'outlier_frac': None,              # which fraction of each class to remove as outliers
	'normalize_log': True,             # use logarithm for normalization
	'use_calibration': False,          # use calibration of probabilities
	'use_rescale_priors': False,       # rescale predictions to match priors
}

make_pretrain(pretrain, train_data, true_labels, **params)

validator = SampleCrossValidator(train_data, true_labels, rounds = 1, test_frac = 0.2, use_data_frac = 1)
optimizer = ParallelGridOptimizer(train_test_func = train_test_NN, validator = validator, use_caching = False, **params
).readygo(topprint = 20, save_fig_basename = name, log_name = name, only_show_top = True)


