
from nnet.base_optimize import optimize_NN


optimize_NN(debug = True, **{
	'dense1_size': 256,
	'dense2_size': 128,
	'learning_rate': 0.0001,
	'momentum': [0.1, 0.9, 0.99, 0.999],
	'dropout1_rate': 0.5,
	'rounds': 3,
})


