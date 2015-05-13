
from nnet.base_optimize import optimize_NN


optimize_NN(debug = True, **{
	'use_calibration': [False, True],
	'use_rescale_priors': [False, True],
})


