
from nnet.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'dense1_size': 256,
	'dense2_size': 128,
	'learning_rate': [0.01, 0.003, 0.001, 0.0003, 0.0001],
	'momentum': 0.9,
	'dropout1_rate': 0.5,
	'rounds': 3,
})


