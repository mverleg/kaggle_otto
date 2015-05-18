
from nnet.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'dense1_nonlinearity': 'rectify',
	'dense1_init': 'glorot_normal',
	'dense1_size': 256,
	'dense2_size': [0, 128],
	'dense3_size': None,
	'learning_rate': [0.0005, 0.0010, 0.0015],
	'learning_rate_scaling': 100,
	'momentum': 0.9,
	'momentum_scaling': 10,
	'dropout1_rate': 0.5,
	'dropout2_rate': None,
	'dropout3_rate': None,
	'weight_decay': [0.00006, 0.00006, 0.00006, 0.00006, 0.00006],
	'max_epochs': 500,
	'auto_stopping': True,
	'extra_feature_count': [0, 163],
	'pretrain': True,
	'save_snapshots_stepsize': 500,
	'rounds': 1,
})


