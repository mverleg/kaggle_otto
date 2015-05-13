
from nnet.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'dense1_size': 512,
	'dense2_size': 512,
	'dense3_size': 256,
	'dropout1_rate': [0, 0.5],
	'weight_decay': [.001, .0001, .00001, 0],
	'max_epochs': 2000,
})


