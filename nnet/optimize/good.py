
from nnet.base_optimize import optimize_NN


optimize_NN(debug = True, **{
	'dense1_size': 256,
	'dense2_size': 256,
	'dense3_size': 128,
	'learning_rate': 0.01,  #todo I might need 10x higher since 1-momentum is 10x smaller
	'learning_rate_scaling': 1000,
	'momentum': 0.99,
	'momentum_scaling': 10,
	'dropout1_rate': 0.5,
	'dropout2_rate': 0.5,
	'weight_decay': 0.0001,  # todo change when regularization results are in
	'max_epochs': 2000,
	'extra_feature_count': 163,
	'pretrain': True,
	#'test_data_confidence': 0.9, #todo change when results are in
})


