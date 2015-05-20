
from nnet.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'dense1_size': 256,
	'dense2_size': 256,
	'dense3_size': None,
	'dropout1_rate': 0.5,
	'extra_feature_count': [7, 17, 37, 67, 107, 157, 217, 287],
	'rounds': 3,
})


