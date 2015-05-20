
from nnet.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'dense1_size': 256,
	'dense2_size': 128,
	'dropout1_rate': 00.5,
	'test_data_confidence': [None, 0.98, 0.95, 0.9, 0.8, 0.6],
	'rounds': 5,
})


