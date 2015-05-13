
from nnet.base_optimize import optimize_NN


optimize_NN(debug = True, **{
	'dense1_size': 512,
	'dense2_size': 512,
	'dense3_size': 256,
	'learning_rate': 0.001,
	'dropout1_rate': [0, 0.5],
	'weight_decay': [0, .00001, .0001, .001, .01],
	'max_epochs': 1500,
	'extra_feature_count': 0,
	'pretrain': True,
})


