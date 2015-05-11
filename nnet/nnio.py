
from cPickle import dump, load
from numpy import savez, load as loadz
from os.path import join
from settings import NNET_STATE_DIR, VERBOSITY


def save_net(net, filepath):
	"""
		Save a neural network to disk in the current state.

		:param net: nolearn/lasagne network.

		This is not guaranteed to work between versions or computers!
	"""
	with open(filepath, 'wb+') as fh:
		dump(obj = net, file = fh, protocol = -1)


def load_net(filepath):
	"""
		Load a neural network from disk.

		:return: nolearn/lasagne network including weights and parameters.
	"""
	with open(filepath, 'r') as fh:
		return load(file = fh)


def save_knowledge(net, filepath):
	"""
		Save the weights and biasses of the neural network to disk.
	"""
	knowledge = [param.get_value() for param in net.get_all_params()]
	savez(filepath, *knowledge)


def load_knowledge(net, filepath):
	"""
		Load the weights and biasses for an already network from disk.
	"""
	reloaded = loadz(filepath)
	knowledge = [(name, reloaded[name]) for name in sorted(reloaded.keys())]
	for (name, values), param in zip(knowledge, net.get_all_params()):
		assert param.get_value().shape == values.shape, 'Loaded data from "{0:s}" does not match shape of network "{1:s}" for "{2:s}": expected {3:s}, got {4:s}'.format(filepath, net, name, param.get_value().shape, values.shape)
		param.set_value(values)


class SnapshotSaver(object):
	def __init__(self, every = 500, base_name = 'net'):
		self.every = every
		self.base_path = join(NNET_STATE_DIR, base_name)

	def __call__(self, nn, train_history):
		epoch = train_history[-1]['epoch']
		if epoch % self.every == 0 or epoch == nn.max_epochs:
			filepath = '{0:s}_{1:d}.net'.format(self.base_path, epoch)
			save_knowledge(nn, filepath)
			if VERBOSITY >= 1:
				print 'saved network to "{0:s}" at iteration {1:d}'.format(filepath, epoch)


