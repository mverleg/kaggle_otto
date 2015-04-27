
"""
	Run the neural network on the Otto data.

	http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
"""

from warnings import filterwarnings
from lasagne.init import Orthogonal, GlorotNormal, GlorotUniform, HeNormal, HeUniform, Sparse, Constant
from numpy import float32
from lasagne.nonlinearities import softmax, tanh, sigmoid, rectify, LeakyRectify
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from theano import shared
from nnet.nnio import SnapshotSaver
from nnet.dynamic import LogarithmicVariable
from settings import NCLASSES, VERBOSITY


filterwarnings('ignore', '.*topo.*')

nonlinearities = {
	'tanh': tanh,
	'sigmoid': sigmoid,
	'rectify': rectify,
	'leaky2': LeakyRectify(leakiness = 0.02),
	'leaky20': LeakyRectify(leakiness = 0.2),
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
		hidden1_size = 60,
		hidden1_nonlinearity = 'tanh',
		hidden1_init = 'orthogonal',
		hidden2_size = None,
		hidden2_nonlinearity = None,
		hidden2_init = 'orthogonal',
		learning_rate_start = 0.001,
		learning_rate_end = 0.00001,
		momentum_start = 0.9,
		momentum_end = 0.999,
		max_epochs = 3000,
		dropout_rate = None,
	):
	"""
		Create the network with the selected parameters.

		:param name: Name for save files
		:param hidden1_size: Number of neurons for first hidden layer
		:param hidden1_nonlinearity: The activation function for the first hidden layer
		:param hidden1_init: The weight initialization for the first hidden layer
		:param learning_rate_start: Start value at first epoch (logarithmic scale)
		:param learning_rate_end: End value at last epoch (logarithmic scale)
		:param momentum_start: Start value at first epoch (logarithmic scale)
		:param momentum_end: End value at last epoch (logarithmic scale)
		:param max_epochs: Total number of epochs
		:param dropout_rate: Percentage of connections dropped each step.
		:return:
	"""

	assert dropout_rate is None, 'Dropout not implemented yet.'
	assert hidden2_size is None and hidden2_nonlinearity is None, 'Second hidden layer not implemented yet.'
	assert hidden1_nonlinearity in nonlinearities.keys(), 'Linearity 1 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), hidden1_nonlinearity)
	assert hidden2_nonlinearity in nonlinearities.keys() + [None], 'Linearity 2 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), hidden2_nonlinearity)
	assert hidden1_init in initializers.keys(), 'Initializer 1 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), hidden1_init)
	assert hidden2_init in initializers.keys() + [None], 'Initializer 2 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), hidden2_init)

	net = NeuralNet(
		layers = [
			('input', layers.InputLayer),
			('dense1', layers.DenseLayer),
			#('dense2', layers.DenseLayer),
			('output', layers.DenseLayer),
		],

		input_shape = (128, 93),  # batch size

		dense1_num_units = hidden1_size,
		dense1_nonlinearity = nonlinearities[hidden1_nonlinearity],
		dense1_W = initializers[hidden1_init],
		dense1_b = Constant(0.),

		# dropout here

		#dense2_num_units = hidden1_size,
		#dense2_nonlinearity = nonlinearities[hidden1_nonlinearity],
		#dense2_W = initializers[hidden1_init],
		#dense2_b = Constant(0.),

		output_nonlinearity = softmax,
		output_num_units = NCLASSES,
		output_W = Orthogonal(),

		update = nesterov_momentum,
		update_learning_rate = shared(float32(learning_rate_start)),
		update_momentum = shared(float32(momentum_start)),

		on_epoch_finished = [
			LogarithmicVariable('update_learning_rate', start = learning_rate_start, stop = learning_rate_end),
			LogarithmicVariable('update_momentum', start = momentum_start, stop = momentum_end),
			SnapshotSaver(every = 100, base_name = name),
		],

		regression = False,
		max_epochs = max_epochs,
		verbose = bool(VERBOSITY),
	)
	return net


