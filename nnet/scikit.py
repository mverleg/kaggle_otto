
"""
	Adapt to scikit learn.

	pre-processing: http://scikit-learn.org/stable/modules/pipeline.html
	optimization: http://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization
	estimator: http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
			   http://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html
	sampling: http://en.wikipedia.org/wiki/Stratified_sampling
"""

from collections import OrderedDict
from functools import partial
from json import load, dump
from os.path import join
from random import random
from sys import stderr
from time import sleep
from numpy import float32, mean, isfinite
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from theano import shared
from sklearn.base import BaseEstimator, ClassifierMixin
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from nnet.weight_decay import WeightDecayObjective, AdaptiveWeightDecay
from validation.optimize import params_name
from nnet.nnio import SnapshotStepSaver, SnapshotEndSaver, save_knowledge, load_knowledge, TrainProgressPlotter, \
	get_knowledge, set_knowledge
from nnet.dynamic import LogarithmicVariable
from nnet.early_stopping import StopWhenOverfitting, StopAfterMinimum, StopNaN, BreakEveryN
from settings import NCLASSES, VERBOSITY, NNET_STATE_DIR, DivergenceError
from lasagne.init import Orthogonal, GlorotNormal, GlorotUniform, HeNormal, HeUniform, Sparse, Constant
from lasagne.nonlinearities import softmax, tanh, sigmoid, rectify, LeakyRectify


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


class NNet(BaseEstimator, ClassifierMixin):

	def __init__(self,
			name = 'nameless_net',          # used for saving, so maybe make it unique
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
			epoch_steps = None,
			dropout0_rate = 0,              # this is the input layer
			dropout1_rate = None,
			dropout2_rate = None,           # inherits dropout1_rate
			dropout3_rate = None,           # inherits dropout2_rate
			weight_decay = 0,
			adaptive_weight_decay = False,
			batch_size = 128,
			output_nonlinearity = 'softmax',
			auto_stopping = True,
			save_snapshots_stepsize = None,
		):
		"""
			Create the network with the selected parameters.

			:param name: Name for save files
			:param dense1_size: Number of neurons for first hidden layer
			:param dense1_nonlinearity: The activation function for the first hidden layer
			:param dense1_init: The weight initialization for the first hidden layer
			:param learning_rate: The (initial) learning rate (how fast the network learns)
			:param learning_rate_scaling: The total factor to gradually decrease the learning rate by
			:param momentum: The (initial) momentum
			:param momentum_scaling: Similar to learning_rate_scaling
			:param max_epochs: Total number of epochs (at most)
			:param dropout1_rate: Percentage of connections dropped each step for first hidden layer
			:param weight_decay: Palatalizes the weights by L2 norm (regularizes but decreases results)
			:param adaptive_weight_decay: Should the weight decay adapt automatically?
			:param batch_size: How many samples to send through the network at a time
			:param auto_stopping: Stop early if the network seems to stop performing well
			:param pretrain: Filepath of the previous weights to start at (or None)
			:return:
		"""

		"""
			Input argument storage: automatically store all locals, which should be exactly the arguments at this point, but storing a little too much is not a big problem.
		"""
		params = locals()
		del params['self']
		self.__dict__.update(params)
		self.parameter_names = sorted(params.keys())

		"""
			Check the parameters and update some defaults (will be done for 'self', no need to store again).
		"""
		self.set_params()

	def init_net(self, feature_count, class_count = NCLASSES, verbosity = VERBOSITY >= 2):
		"""
			Initialize the network (needs to be done when data is available in order to set dimensions).
		"""
		if VERBOSITY >= 1:
			print 'initializing network {0:s} {1:d}x{2:d}x{3:d}'.format(self.name, self.dense1_size or 0, self.dense2_size or 0, self.dense3_size or 0)
			if VERBOSITY >= 2:
				print 'parameters: ' + ', '.join('{0:s} = {1:}'.format(k, v) for k,v in self.get_params(deep = False).items())
		self.feature_count = feature_count
		self.class_count = class_count

		"""
			Create the layers and their settings.
		"""
		self.layers = [
			('input', InputLayer),
		]
		self.params = {
			'dense1_num_units': self.dense1_size,
			'dense1_nonlinearity': nonlinearities[self.dense1_nonlinearity],
			'dense1_W': initializers[self.dense1_init],
			'dense1_b': Constant(0.),
		}
		if self.dropout0_rate:
			self.layers += [('dropout0', DropoutLayer)]
			self.params['dropout0_p'] = self.dropout0_rate
		self.layers += [('dense1', DenseLayer),]
		if self.dropout1_rate:
			self.layers += [('dropout1', DropoutLayer)]
			self.params['dropout1_p'] = self.dropout1_rate
		if self.dense2_size:
			self.layers += [('dense2', DenseLayer)]
			self.params.update({
				'dense2_num_units': self.dense2_size,
				'dense2_nonlinearity': nonlinearities[self.dense2_nonlinearity],
				'dense2_W': initializers[self.dense2_init],
				'dense2_b': Constant(0.),
			})
		else:
			assert not self.dense3_size, 'There cannot be a third dense layer without a second one'
		if self.dropout2_rate:
			assert self.dense2_size is not None, 'There cannot be a second dropout layer without a second dense layer.'
			self.layers += [('dropout2', DropoutLayer)]
			self.params['dropout2_p'] = self.dropout2_rate
		if self.dense3_size:
			self.layers += [('dense3', DenseLayer)]
			self.params.update({
				'dense3_num_units': self.dense3_size,
				'dense3_nonlinearity': nonlinearities[self.dense3_nonlinearity],
				'dense3_W': initializers[self.dense3_init],
				'dense3_b': Constant(0.),
			})
		if self.dropout3_rate:
			assert self.dense2_size is not None, 'There cannot be a third dropout layer without a third dense layer.'
			self.layers += [('dropout3', DropoutLayer)]
			self.params['dropout3_p'] = self.dropout2_rate
		self.layers += [('output', DenseLayer)]
		self.params.update({
			'output_nonlinearity': nonlinearities[self.output_nonlinearity],
			'output_W': Orthogonal(),
		})

		"""
			Create meta parameters and special handlers.
		"""
		if VERBOSITY >= 3:
			print 'learning rate: {0:.6f} -> {1:.6f}'.format(abs(self.learning_rate), abs(self.learning_rate) / float(self.learning_rate_scaling))
			print 'momentum:      {0:.6f} -> {1:.6f}'.format(abs(self.momentum), 1 - ((1 - abs(self.momentum)) / float(self.momentum_scaling)))
		self.step_handlers = [
			LogarithmicVariable('update_learning_rate', start = abs(self.learning_rate), stop = abs(self.learning_rate) / float(self.learning_rate_scaling)),
			LogarithmicVariable('update_momentum', start = abs(self.momentum), stop = 1 - ((1 - abs(self.momentum)) / float(self.momentum_scaling))),
			StopNaN(),
		]
		self.end_handlers = [
			SnapshotEndSaver(base_name = self.name),
			TrainProgressPlotter(base_name = self.name),
		]
		snapshot_name = 'nn_' + params_name(self.params, prefix = self.name)[0]
		if self.save_snapshots_stepsize:
			self.step_handlers += [
				SnapshotStepSaver(every = self.save_snapshots_stepsize, base_name = snapshot_name),
			]
		if self.auto_stopping:
			self.step_handlers += [
				StopWhenOverfitting(loss_fraction = 0.8, base_name = snapshot_name),
				StopAfterMinimum(patience = 40, base_name = self.name),
			]
		weight_decay = shared(float32(abs(self.weight_decay)), 'weight_decay')
		if self.adaptive_weight_decay:
			self.step_handlers += [
				AdaptiveWeightDecay(weight_decay),
			]
		if self.epoch_steps:
			self.step_handlers += [
				BreakEveryN(self.epoch_steps),
			]

		"""
			Create the actual nolearn network with information from __init__.
		"""
		self.net = NeuralNet(

			layers = self.layers,

			objective = partial(WeightDecayObjective, weight_decay = weight_decay),

			input_shape = (None, feature_count),
			output_num_units = class_count,

			update = nesterov_momentum,
			update_learning_rate = shared(float32(self.learning_rate)),
			update_momentum = shared(float(self.weight_decay)),

			on_epoch_finished = self.step_handlers,
			on_training_finished = self.end_handlers,

			regression = False,
			max_epochs = self.max_epochs,
			verbose = verbosity,

			batch_iterator_train = BatchIterator(batch_size = self.batch_size),
			batch_iterator_test = BatchIterator(batch_size = self.batch_size),

			eval_size = 0.1,

			#custom_score = ('custom_loss', categorical_crossentropy),

			**self.params
		)

		self.net.initialize()

		return self.net

	def get_params(self, deep = True):
		return OrderedDict((name, getattr(self, name)) for name in self.parameter_names)

	def set_params(self, **params):
		"""
			Set all the parameters.
		"""
		for name, val in params.items():
			assert name in self.parameter_names, '"{0:s}" is not a valid parameter name (known parameters: "{1:s}")'.format(name, '", "'.join(self.parameter_names))
			setattr(self, name, val)

		"""
			Arguments checks.
		"""
		assert self.dropout1_rate is None or 0 <= self.dropout1_rate < 1, 'Dropout rate 1 should be a value between 0 and 1 (value: {0})'.format(self.dropout1_rate)
		assert self.dropout2_rate is None or 0 <= self.dropout2_rate < 1, 'Dropout rate 2 should be a value between 0 and 1, or None for inheritance (value: {0})'.format(self.dropout2_rate)
		assert self.dropout3_rate is None or 0 <= self.dropout3_rate < 1, 'Dropout rate 3 should be a value between 0 and 1, or None for inheritance (value: {0})'.format(self.dropout3_rate)
		assert self.dense1_nonlinearity in nonlinearities.keys(), 'Linearity 1 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), self.dense1_nonlinearity)
		assert self.dense2_nonlinearity in nonlinearities.keys() + [None], 'Linearity 2 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), self.dense2_nonlinearity)
		assert self.dense3_nonlinearity in nonlinearities.keys() + [None], 'Linearity 3 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), self.dense3_nonlinearity)
		assert self.dense1_init in initializers.keys(), 'Initializer 1 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), self.dense1_init)
		assert self.dense2_init in initializers.keys() + [None], 'Initializer 2 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), self.dense2_init)
		assert self.dense3_init in initializers.keys() + [None], 'Initializer 3 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), self.dense3_init)

		"""
			Argument defaults.
		"""
		if self.dense2_nonlinearity is None:
			self.dense2_nonlinearity = self.dense1_nonlinearity
		if self.dense2_init is None:
			self.dense2_init = self.dense1_init
		if self.dense3_nonlinearity is None:
			self.dense3_nonlinearity = self.dense2_nonlinearity
		if self.dense3_init is None:
			self.dense3_init = self.dense2_init
		if self.dropout2_rate is None and self.dense2_size:
			self.dropout2_rate = self.dropout1_rate
		if self.dropout3_rate is None and self.dense3_size:
			self.dropout3_rate = self.dropout2_rate

	def fit(self, X, y, random_sleep = None):
		if random_sleep:
			sleep(random_sleep * random())  # this is to prevent compiler lock problems
		labels = y - y.min()
		self.init_net(feature_count = X.shape[1], class_count = labels.max() + 1)
		net = self.net.fit(X, labels)
		self.save()
		return net

	def interrupted_fit(self, X, y):
		""" DEPRECATED """
		labels = y - y.min()
		self.init_net(feature_count = X.shape[1], class_count = labels.max() + 1)
		knowledge = get_knowledge(self.net)
		for epoch in range(0, self.max_epochs, self.epoch_steps):
			#todo: so now dynamic variables don't work
			set_knowledge(self.net, knowledge)
			self.init_net(feature_count = X.shape[1], class_count = labels.max() + 1)
			print 'epoch {0:d}: learning {1:d} epochs'.format(epoch, self.epoch_steps)
			self.net.fit(X, labels)
			ratio = mean([d['valid_loss'] for d in self.net._train_history[-self.epoch_steps:]]) / \
					mean([d['train_loss'] for d in self.net._train_history[-self.epoch_steps:]])
			if ratio < 0.85:
				self.weight_decay *= 1.3
			if ratio > 0.95:
				self.weight_decay /= 1.2
			self.init_net(feature_count = X.shape[1], class_count = labels.max() + 1)
			knowledge = get_knowledge(self.net)
		exit()
		net = self.net.fit(X, labels)
		self.save()
		return net

	def predict_proba(self, X):
		probs = self.net.predict_proba(X)
		if not isfinite(probs).sum():
			errmsg = 'network "{0:s}" predicted infinite/NaN probabilities'.format(self.name)
			stderr.write(errmsg)
			raise DivergenceError(errmsg)
		return probs

	def predict(self, X):
		return self.net.predict(X)

	def score(self, X, y, **kwargs):
		return self.net.score(X, y)

	def save(self, filepath = None):
		assert hasattr(self, 'net'), 'Cannot save a network that is not initialized; .fit(X, y) something first [or use net.initialize(..) for random initialization].'
		parameters = self.get_params(deep = False)
		filepath = filepath or join(NNET_STATE_DIR, self.name)
		if VERBOSITY >= 1:
			print 'saving network to "{0:s}.net.npz|json"'.format(filepath)
		with open(filepath + '.net.json', 'w+') as fh:
			dump([parameters, self.feature_count, self.class_count], fp = fh)
		save_knowledge(self.net, filepath + '.net.npz')

	@classmethod
	def load(cls, filepath = None, name = None):
		"""
			:param filepath: The base path (without extension) to load the file from, OR:
			:param name: The name of the network to load (if filename is not given)
			:return: The loaded network
		"""
		filepath = filepath or join(NNET_STATE_DIR, name)
		if VERBOSITY >= 1:
			print 'loading network from "{0:s}.net.npz|json"'.format(filepath)
		with open(filepath + '.net.json', 'r') as fh:
			[parameters, feature_count, class_count] = load(fp = fh)
		nnet = cls(**parameters)
		nnet.init_net(feature_count = feature_count, class_count = class_count)
		load_knowledge(nnet.net, filepath + '.net.npz')
		return nnet


