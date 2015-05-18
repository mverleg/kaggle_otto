
from nnet.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'dense1_size': 256,
	'dense2_size': 256,
	'dense3_size': 128,
	'learning_rate': [0.015, 0.008,0.004],
	'learning_rate_scaling': 500,
	'momentum': 0.99,
	'momentum_scaling': 10,
	'dropout1_rate': 0.5,
	'dropout2_rate': 0.5,
	'weight_decay': [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01],
	'max_epochs': 1500,
	'extra_feature_count': 163,
	'pretrain': True,
	'save_snapshots_stepsize': 1500,
	'rounds': 3,
})



