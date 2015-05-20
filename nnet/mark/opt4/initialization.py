
from nnet.base_optimize import optimize_NN


optimize_NN(debug = False, **{
	'dense1_size': 128,
	'dense2_size': 64,
	'dense1_init': ['orthogonal', 'sparse', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
	'rounds': 3,
})


