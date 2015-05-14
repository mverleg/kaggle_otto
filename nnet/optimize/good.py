
from nnet.base_optimize import optimize_NN


optimize_NN(debug = True, **{
	'dense1_size': 512,
	'dense2_size': 512,
	'dense3_size': 256,
	'learning_rate': 0.01,
	'learning_rate_scaling': 1000,
	'momentum': 0.99,
	'momentum_scaling': 10,
	'dropout1_rate': 0.5,
	'dropout2_rate': 0.5,
	'weight_decay': 0.0001,  # todo
	'max_epochs': 2000,
	'extra_feature_count': 163,
	'pretrain': True,
})


