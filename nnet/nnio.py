
from cPickle import dump, load
from os.path import dirname, join
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


class SnapshotSaver(object):
	def __init__(self, every = 500, base_name = 'net'):
		self.every = every
		self.base_path = join(NNET_STATE_DIR, base_name)

	def __call__(self, nn, train_history):
		epoch = train_history[-1]['epoch']
		if epoch % self.every == 0 or epoch == nn.max_epochs:
			filepath = '{0:s}_{1:d}.net'.format(self.base_path, epoch)
			save_net(nn, filepath)
			if VERBOSITY >= 1:
				print 'saved network to "{0:s}" at iteration {1:d}'.format(filepath, epoch)


