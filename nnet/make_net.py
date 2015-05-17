
"""
	Run the neural network on the Otto data.

	http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
"""

from functools import partial
from sys import setrecursionlimit
from os.path import isfile
from nnet.weight_decay import WeightDecayObjective
from validation.optimize import params_name
from warnings import filterwarnings
from lasagne.init import Orthogonal, GlorotNormal, GlorotUniform, HeNormal, HeUniform, Sparse, Constant
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from numpy import float32
from lasagne.nonlinearities import softmax, tanh, sigmoid, rectify, LeakyRectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from theano import shared
from nnet.nnio import SnapshotSaver, load_knowledge
from nnet.dynamic import LogarithmicVariable
from nnet.early_stopping import StopWhenOverfitting, StopAfterMinimum, StopNaN
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
		NFEATS,
		name = 'hidden1_size',
		dense1_size = 60,
		dense1_nonlinearity = 'tanh',
		dense1_init = 'orthogonal',
		dense2_size = None,
		dense2_nonlinearity = None,     # inherits dense1
		dense2_init = None,             # inherits dense1
		dense3_size = None,
		dense3_nonlinearity = None,     # inherits dense2
		dense3_init = None,             # inherits dense2
		learning_rate = 0.001,
		learning_rate_scaling = 100,
		momentum = 0.9,
		momentum_scaling = 100,
		max_epochs = 3000,
		dropout1_rate = None,
		dropout2_rate = None,           # inherits dropout1_rate
		weight_decay = 0,
		output_nonlinearity = 'softmax',
		auto_stopping = True,
		pretrain = False,
		save_snapshots_stepsize = None,
		verbosity = VERBOSITY >= 2,
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
		:param max_epochs: Total number of epochs (at most)
		:param dropout1_rate: Percentage of connections dropped each step.
		:param weight_decay: Constrain the weights by L2 norm.
		:param auto_stopping: Stop early if the network seems to stop performing well.
		:param pretrain: Filepath of the previous weights to start at (or None).
		:return:
	"""
	"""
		Initial arguments checks and defaults.
	"""
	assert dropout1_rate is None or 0 <= dropout1_rate < 1, 'Dropout rate 1 should be a value between 0 and 1, or None for no dropout'
	assert dropout2_rate is None or 0 <= dropout1_rate < 1, 'Dropout rate 2 should be a value between 0 and 1, or None for no dropout'
	assert dense1_nonlinearity in nonlinearities.keys(), 'Linearity 1 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), dense1_nonlinearity)
	assert dense2_nonlinearity in nonlinearities.keys() + [None], 'Linearity 2 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), dense2_nonlinearity)
	assert dense3_nonlinearity in nonlinearities.keys() + [None], 'Linearity 3 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), dense3_nonlinearity)
	assert dense1_init in initializers.keys(), 'Initializer 1 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), dense1_init)
	assert dense2_init in initializers.keys() + [None], 'Initializer 2 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), dense2_init)
	assert dense3_init in initializers.keys() + [None], 'Initializer 3 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), dense3_init)

	if dense2_nonlinearity is None:
		dense2_nonlinearity = dense1_nonlinearity
	if dense2_init is None:
		dense2_init = dense1_init
	if dense3_nonlinearity is None:
		dense3_nonlinearity = dense2_nonlinearity
	if dense3_init is None:
		dense3_init = dense2_init
	if dropout2_rate is None and dense3_size:
		dropout2_rate = dropout1_rate

	"""
		Create the layers and their settings.
	"""
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
	else:
		assert dense3_size is None, 'There cannot be a third dense layer without a second one'
	if dropout2_rate:
		assert dense2_size is not None, 'There cannot be a second dropout layer without a second dense layer.'
		layers += [('dropout2', DropoutLayer)]
		params['dropout2_p'] = dropout2_rate
	if dense3_size:
		layers += [('dense3', DenseLayer)]
		params.update({
			'dense3_num_units': dense3_size,
			'dense3_nonlinearity': nonlinearities[dense3_nonlinearity],
			'dense3_W': initializers[dense3_init],
			'dense3_b': Constant(0.),
		})
	layers += [('output', DenseLayer)]

	"""
		Create meta parameters and special handlers.
	"""
	if VERBOSITY >= 1:
		print 'learning rate: {0:.6f} -> {1:.6f}'.format(learning_rate, learning_rate / float(learning_rate_scaling))
		print 'momentum:      {0:.6f} -> {1:.6f}'.format(momentum, 1 - ((1 - momentum) / float(momentum_scaling)))
	handlers = [
		LogarithmicVariable('update_learning_rate', start = learning_rate, stop = learning_rate / float(learning_rate_scaling)),
		LogarithmicVariable('update_momentum', start = momentum, stop = 1 - ((1 - momentum) / float(momentum_scaling))),
		StopNaN(),
	]
	snapshot_name = 'nn_' + params_name(params, prefix = name)[0]
	if save_snapshots_stepsize:
		handlers += [
			SnapshotSaver(every = save_snapshots_stepsize, base_name = snapshot_name),
		]
	if auto_stopping:
		handlers += [
			StopWhenOverfitting(loss_fraction = 0.8, base_name = snapshot_name),
			StopAfterMinimum(patience = 40, base_name = name),
		]

	"""
		Create the actual nolearn network with above information.
	"""
	net = NeuralNet(

		layers = layers,

		objective = partial(WeightDecayObjective, decay = weight_decay),

		input_shape = (None, NFEATS),

		dense1_num_units = dense1_size,
		dense1_nonlinearity = nonlinearities[dense1_nonlinearity],
		dense1_W = initializers[dense1_init],
		dense1_b = Constant(0.),

		output_nonlinearity = nonlinearities[output_nonlinearity],
		output_num_units = NCLASSES,
		output_W = Orthogonal(),

		update = nesterov_momentum,
		update_learning_rate = shared(float32(learning_rate)),
		update_momentum = shared(float32(momentum)),

		on_epoch_finished = handlers,

		regression = False,
		max_epochs = max_epochs,
		verbose = verbosity,

		**params
	)

	net.initialize()

	"""
		Load weights from earlier training (by name, no auto-choosing).
	"""
	if pretrain:
		assert isfile(pretrain), 'Pre-train file "{0:s}" not found'.format(pretrain)
		load_knowledge(net, pretrain)

	return net


