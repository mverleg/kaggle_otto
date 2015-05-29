
from nnet.oldstyle.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'use_calibration': [False, True],
	'use_rescale_priors': [False, True],
})


