
from nnet.base_optimize import optimize_NN
from nnet.train_test import train_test_NN


def train_test_NN_scale_lr(train, labels, test, learning_rate = None, momentum = None, **kwargs):
	print 'rescaling learning rate: {0:.6f} -> {1:.6f} (for momentum {2:.6f})'.format(learning_rate, learning_rate / (1 - momentum), momentum)
	return train_test_NN(train, labels, test, learning_rate = learning_rate * (1 - momentum), momentum = momentum, **kwargs)


optimize_NN(debug = True, train_test_func = train_test_NN_scale_lr, **{
	'dense1_size': 256,
	'dense2_size': 128,
	'learning_rate': 0.01,
	'learning_rate_scaling': 1000,
	'momentum': [0.1, 0.9, 0.99, 0.999],
	'momentum_scaling': 100,
	'dropout1_rate': 0.5,
	'rounds': 3,
})


