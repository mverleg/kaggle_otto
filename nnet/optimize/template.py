
from nnet.base_optimize import optimize


optimize(debug = True, **{
	'dense1_size': 556,
	'dense2_size': None,
	'dense3_size': None,
	'learning_rate': 0.0001,
	'dropout1_rate': 0,
	'max_epochs': 1,
	'extra_feature_count': 20,
	'extra_feature_seed': 0,
	'pretrain': False,
})

# extra_feature_count converges normally up to some point (19 normally, 18 with double, 5[min] with triple)
# but at some point diverges to NaN at the first iteration
# does seed have any effect? -> it didn't at first because of bug but it does now
# learning rate seems to have little effect on this
# todo: there is apparently something wrong with feature nr 20 at seed 0?
#


