
from nnet.base_optimize import optimize


optimize(debug = True, **{
	'dense1_size': 256,
	'dense2_size': None,
	'dense3_size': None,
	'learning_rate': 0.001,
	'dropout1_rate': 0,
	'max_epochs': 20,
	'extra_feature_count': 13,
	'pretrain': True,
})


