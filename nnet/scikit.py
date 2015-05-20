
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
from json import load
from json import dump
from os.path import join
from theano.tensor.nnet import categorical_crossentropy
from nnet.weight_decay import WeightDecayObjective
from lasagne.init import Orthogonal
from numpy import float32
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from theano import shared
from sklearn.base import BaseEstimator, ClassifierMixin
from nnet.make_net import nonlinearities
from nnet.make_net import initializers
from validation.optimize import params_name
from lasagne.init import Constant
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from nnet.nnio import SnapshotStepSaver, SnapshotEndSaver, save_knowledge, load_knowledge
from nnet.dynamic import LogarithmicVariable
from nnet.early_stopping import StopWhenOverfitting, StopAfterMinimum, StopNaN
from settings import NCLASSES, VERBOSITY, NNET_STATE_DIR
from copy import copy


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
			dropout1_rate = None,
			dropout2_rate = None,           # inherits dropout1_rate
			dropout3_rate = None,
			weight_decay = 0,
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
			Initial arguments checks.
		"""
		assert dropout1_rate is None or 0 <= dropout1_rate < 1, 'Dropout rate 1 should be a value between 0 and 1'
		assert dropout2_rate is None or 0 <= dropout1_rate < 1, 'Dropout rate 2 should be a value between 0 and 1, or None for inheritance'
		assert dropout3_rate is None or 0 <= dropout1_rate < 1, 'Dropout rate 3 should be a value between 0 and 1, or None for inheritance'
		assert dense1_nonlinearity in nonlinearities.keys(), 'Linearity 1 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), dense1_nonlinearity)
		assert dense2_nonlinearity in nonlinearities.keys() + [None], 'Linearity 2 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), dense2_nonlinearity)
		assert dense3_nonlinearity in nonlinearities.keys() + [None], 'Linearity 3 should be one of "{0}", got "{1}" instead.'.format('", "'.join(nonlinearities.keys()), dense3_nonlinearity)
		assert dense1_init in initializers.keys(), 'Initializer 1 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), dense1_init)
		assert dense2_init in initializers.keys() + [None], 'Initializer 2 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), dense2_init)
		assert dense3_init in initializers.keys() + [None], 'Initializer 3 should be one of "{0}", got "{1}" instead.'.format('", "'.join(initializers.keys()), dense3_init)

		"""
			Input argument defaults.
		"""
		if dense2_nonlinearity is None:
			dense2_nonlinearity = dense1_nonlinearity
		if dense2_init is None:
			dense2_init = dense1_init
		if dense3_nonlinearity is None:
			dense3_nonlinearity = dense2_nonlinearity
		if dense3_init is None:
			dense3_init = dense2_init
		if dropout2_rate is None and dense2_size:
			dropout2_rate = dropout1_rate
		if dropout3_rate is None and dense3_size:
			dropout3_rate = dropout2_rate

		"""
			Input argument storage: automatically store all locals, which should be exactly the arguments at this point, but storing a little too much is not a big problem.
		"""
		self.__dict__.update(locals())
		self.parameter_names = sorted(copy(locals().keys()))
		self.parameter_names.remove('self')

	def init_net(self, feature_count, class_count = NCLASSES, verbosity = VERBOSITY >= 2):
		"""
			Initialize the network (needs to be done when data is available in order to set dimensions).
		"""
		if VERBOSITY >= 1:
			print 'initializing network {0:s} {1:d}x{2:d}x{3:d}'.format(self.name, self.dense1_size or 0, self.dense2_size or 0, self.dense3_size or 0)
			if VERBOSITY >= 2:
				print 'parameters: ' + ', '.join('{0:s} = {1:}'.format(k, v) for k,v in self.get_params(deep = False).items())
		"""
			Create the layers and their settings.
		"""
		self.layers = [
			('input', InputLayer),
			('dense1', DenseLayer),
		]
		self.params = {
			'dense1_num_units': self.dense1_size,
			'dense1_nonlinearity': nonlinearities[self.dense1_nonlinearity],
			'dense1_W': initializers[self.dense1_init],
			'dense1_b': Constant(0.),
		}
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
			assert self.dense3_size is None, 'There cannot be a third dense layer without a second one'
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
			print 'learning rate: {0:.6f} -> {1:.6f}'.format(self.learning_rate, self.learning_rate / float(self.learning_rate_scaling))
			print 'momentum:      {0:.6f} -> {1:.6f}'.format(self.momentum, 1 - ((1 - self.momentum) / float(self.momentum_scaling)))
		self.handlers = [
			LogarithmicVariable('update_learning_rate', start = self.learning_rate, stop = self.learning_rate / float(self.learning_rate_scaling)),
			LogarithmicVariable('update_momentum', start = self.momentum, stop = 1 - ((1 - self.momentum) / float(self.momentum_scaling))),
			StopNaN(),
		]
		snapshot_name = 'nn_' + params_name(self.params, prefix = self.name)[0]
		if self.save_snapshots_stepsize:
			self.handlers += [
				SnapshotStepSaver(every = self.save_snapshots_stepsize, base_name = snapshot_name),
			]
		if self.auto_stopping:
			self.handlers += [
				StopWhenOverfitting(loss_fraction = 0.8, base_name = snapshot_name),
				StopAfterMinimum(patience = 40, base_name = self.name),
			]

		"""
			Create the actual nolearn network with information from __init__.
		"""
		self.net = NeuralNet(

			layers = self.layers,

			objective = partial(WeightDecayObjective, decay = self.weight_decay),

			input_shape = (None, feature_count),
			output_num_units = class_count,

			update = nesterov_momentum,
			update_learning_rate = shared(float32(self.learning_rate)),
			update_momentum = shared(float32(self.momentum)),

			on_epoch_finished = self.handlers,
			on_training_finished = [SnapshotEndSaver(base_name = self.name)],

			regression = False,
			max_epochs = self.max_epochs,
			verbose = verbosity,

			batch_iterator_train = BatchIterator(batch_size = self.batch_size),
			batch_iterator_test = BatchIterator(batch_size = self.batch_size),

			eval_size = 0.1,

			#custom_loss = categorical_crossentropy, # todo

			**self.params
		)

		self.net.initialize()

		return self.net

	def get_params(self, deep = True):
		return OrderedDict((name, getattr(self, name)) for name in self.parameter_names)

	def set_params(self, **params):
		for name, val in params.items():
			assert name in self.parameter_names, '"{0:s}" is not a valid parameter name (known parameters: "{1:s}")'.format(name, '", "'.join(self.parameter_names))
			setattr(self, name, val)

	def fit(self, X, y):
		labels = y - y.min()
		self.init_net(feature_count = X.shape[1], class_count = labels.max() + 1)
		net = self.net.fit(X, labels)
		self.save()
		return net

	def predict_proba(self, X):
		return self.net.predict_proba(X)

	def predict(self, X):
		return self.net.predict(X)

	def score(self, X, y, **kwargs):
		return self.net.score(X, y)

	def save(self, filepath = None):
		parameters = self.get_params(deep = False)
		filepath = filepath or join(NNET_STATE_DIR, self.name)
		if VERBOSITY >= 1:
			print 'saving network to "{0:s}.net.npz/.net.json"'.format(filepath)
		with open(filepath + '.net.json', 'w+') as fh:
			dump(parameters, fp = fh)
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
			print 'loading network from "{0:s}.net.npz/.net.json"'.format(filepath)
		with open(filepath + '.json', 'w+') as fh:
			parameters = load(fp = fh)
		net = cls(**parameters)
		load_knowledge(net, filepath + '.net.npz')
		return net


