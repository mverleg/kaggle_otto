
from nnet.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'dense1_size': 256,
	'dense2_size': [64, 128, 256],
	'dense3_size': [None, 64],
	'learning_rate': [0.015, 0.008,0.004],
	'learning_rate_scaling': 500,
	'momentum': 0.99,
	'momentum_scaling': 10,
	'dropout1_rate': [0, 0.5],
	'dropout2_rate': None,
	'weight_decay': 0,
	'max_epochs': 1500,
	'extra_feature_count': 163,
	'pretrain': False,
	'save_snapshots_stepsize': 1500,
	'rounds': 5,
})



