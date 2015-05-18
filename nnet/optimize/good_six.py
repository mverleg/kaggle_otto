
from nnet.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'dense1_size': 256,
	'dense2_size': 128,
	'dense3_size': None,
	'learning_rate': 0.005,
	'learning_rate_scaling': 5000,
	'momentum': 0.9,
	'momentum_scaling': 100,
	'dropout1_rate': 0.5,
	'dropout2_rate': 0,
	'dropout3_rate': 0,
	'weight_decay': 0,
	'max_epochs': 1500,
	'extra_feature_count': [0, 163],
	'pretrain': True,
	'save_snapshots_stepsize': 1500,
	'rounds': 3,
})


