
from nnet.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'dense1_size': 128,
	'dense2_size': 64,
	'dense1_nonlinearity': ['tanh', 'sigmoid', 'rectify', 'leaky2', 'leaky20', 'softmax'],
	'rounds': 3,
})


