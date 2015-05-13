
from nnet.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'outlier_method': ['OCSVM', 'EE'],
	'outlier_frac': [0, 0.02, 0.5, 0.12, 0.2],
	'rounds': 3,
})


