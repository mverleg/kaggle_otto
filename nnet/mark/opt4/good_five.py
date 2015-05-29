
from nnet.oldstyle.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'dense1_size': 256,
	'dense2_size': 256,
	'dense3_size': [64, 128],
	'learning_rate': 0.005,
	'learning_rate_scaling': 100,
	'momentum': 0.99,
	'momentum_scaling': 10,
	'dropout1_rate': 0.5,
	'dropout2_rate': 0.5,
	'dropout3_rate': [0, 0.5],
	'weight_decay': [0.0002, 0.0005, 0.001, 0.002, 0.004],
	'max_epochs': 1500,
	'extra_feature_count': [0, 163],
	'pretrain': False,
	'save_snapshots_stepsize': 1500,
	'rounds': 3,
})


