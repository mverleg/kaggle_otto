
"""
	Run the neural network on the Otto data.

	http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
"""

from sys import setrecursionlimit
from warnings import filterwarnings
from lasagne.init import Orthogonal, GlorotNormal, GlorotUniform, HeNormal, HeUniform, Sparse, Constant
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from numpy import float32
from lasagne.nonlinearities import softmax, tanh, sigmoid, rectify, LeakyRectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from theano import shared
from nnet.nnio import SnapshotSaver
from nnet.dynamic import LogarithmicVariable
from nnet.early_stopping import StopWhenOverfitting, StopAfterMinimum
from settings import NCLASSES, VERBOSITY


filterwarnings('ignore', '.*topo.*')
setrecursionlimit(10000)

nonlinearities = {
	'tanh': tanh,
	'sigmoid': sigmoid,
	'rectify': rectify,
	'leaky2': LeakyRectify(leakiness = 0.02),
	'leaky20': LeakyRectify(leakiness = 0.2),
	'softmax': softmax,
}

initializers = {
	'orthogonal': Orthogonal(),
	'sparse': Sparse(),
	'glorot_normal': GlorotNormal(),
	'glorot_uniform': GlorotUniform(),
	'he_normal': HeNormal(),
	'he_uniform': HeUniform(),
}


def make_net(
		name = 'hidden1_size',
		dense1_size = 60,
		dense1_nonlinearity = 'tanh',
		dense1_init = 'orthogonal',
		dense2_size = None,
		dense2_nonlinearity = 'tanh',
		dense2_init = 'he_normal',
		learning_rate = 0.001,
		learning_rate_scaling = 100,
		momentum = 0.9,
		momentum_scaling = 100,
		max_epochs = 3000,
		dropout1_rate = None,
		dropout2_rate = None,
		weight_decay = 0,
		output_nonlinearity = 'softmax',
		verbosity = VERBOSITY >= 2,
		auto_stopping = True,
	):
	"""
		Create the network with the selected parameters.

		:param name: Name for save files
		:param dense1_size: Number of neurons for first hidden layer
		:param dense1_nonlinearity: The activation function for the first hidden layer
		:param dense1_init: The weight initialization for the first hidden layer
		:param learning_rate_start: Start value at first epoch (logarithmic scale)
		:param learning_rate_end: End value at last epoch (logarithmic scale)
		:param momentum_start: Start value at first epoch (logarithmic scale)
		:param momentum_end: End value at last epoch (logarithmic scale)
		:param max_epochs: Total number of epochs
		:param dropout1_rate: Percentage of connections dropped each step.
		:return:
	"""

	assert dropout1_rate is None or 0 <= dropout1_rate < 1, 'Dropout rate #1 should be a value between 0 and 1, or None for no dropout'
	assert dropout2_rate is None or 0 <= dropout1_rate < 1, 'Dropout rate #2 should be a value between 0 and 1, or None for no dropout'
	assert dense1_nonlinearity in nonlinearities.keys(), 'Linearity 1 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), dense1_nonlinearity)
	assert dense2_nonlinearity in nonlinearities.keys() + [None], 'Linearity 2 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), dense2_nonlinearity)
	assert dense1_init in initializers.keys(), 'Initializer 1 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), dense1_init)
	assert dense2_init in initializers.keys() + [None], 'Initializer 2 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), dense2_init)
	assert weight_decay == 0, 'Weight decay doesn\'t fully work in Lasagne/nolearn yet. More info https://github.com/dnouri/nolearn/pull/53' # and https://groups.google.com/forum/#!topic/lasagne-users/sUY7K4diHhY

	params = {}
	layers = [
		('input', InputLayer),
		('dense1', DenseLayer),
	]
	if dropout1_rate:
		layers += [('dropout1', DropoutLayer)]
		params['dropout1_p'] = dropout1_rate
	if dense2_size:
		layers += [('dense2', DenseLayer)]
		params.update({
			'dense2_num_units': dense2_size,
			'dense2_nonlinearity': nonlinearities[dense2_nonlinearity],
			'dense2_W': initializers[dense2_init],
			'dense2_b': Constant(0.),
		})
	if dropout2_rate:
		if not dense2_size:
			raise AssertionError('There cannot be a second dropout layer without a second dense layer.')
		layers += [('dropout2', DropoutLayer)]
		params['dropout2_p'] = dropout2_rate
	layers += [('output', DenseLayer)]

	handlers = [
		LogarithmicVariable('update_learning_rate', start = learning_rate_start, stop = learning_rate_end),
		LogarithmicVariable('update_momentum', start = momentum_start, stop = momentum_end),
	]
	if auto_stopping:
		handlers += [
			SnapshotSaver(every = 100, base_name = name),
			StopWhenOverfitting(loss_fraction = 0.8, base_name = name),
			StopAfterMinimum(patience = 70, base_name = name),
		]

	net = NeuralNet(
		layers = layers,

		input_shape = (None, 93),  # batch size

		dense1_num_units = dense1_size,
		dense1_nonlinearity = nonlinearities[dense1_nonlinearity],
		dense1_W = initializers[dense1_init],
		dense1_b = Constant(0.),

		output_nonlinearity = nonlinearities[output_nonlinearity],
		output_num_units = NCLASSES,
		output_W = Orthogonal(),

		update = nesterov_momentum,
		update_learning_rate = shared(float32(learning_rate_start)),
		update_momentum = shared(float32(momentum_start)),

		#regularization = l2,
		#regularization_rate = weight_decay,

		on_epoch_finished = handlers,

		regression = False,
		max_epochs = max_epochs,
		verbose = verbosity,

		**params
	)
	return net


