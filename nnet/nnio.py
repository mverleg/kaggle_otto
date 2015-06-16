
from cPickle import dump, load
from numpy import savez, load as loadz
from os.path import join
from sklearn.metrics.scorer import _ProbaScorer, log_loss
from settings import NNET_STATE_DIR, VERBOSITY, AUTO_IMAGES_DIR


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


def get_knowledge(net):
	return [param.get_value() for param in net.get_all_params()]


def set_knowledge(net, knowledge):
	for (name, values), param in zip(knowledge, net.get_all_params()):
		assert param.get_value().shape == values.shape, 'Data does not match shape of network "{1:s}" for "{2:s}": expected {3:s}, got {4:s}'.format(net, name, param.get_value().shape, values.shape)
		param.set_value(values)


def save_knowledge(net, filepath):
	"""
		Save the weights and biasses of the neural network to disk.
	"""
	knowledge = get_knowledge(net)
	savez(filepath, *knowledge)


def load_knowledge(net, filepath):
	"""
		Load the weights and biasses for an already network from disk.
	"""
	reloaded = loadz(filepath)
	knowledge = [(name, reloaded[name]) for name in sorted(reloaded.keys())]
	set_knowledge(net, knowledge)


class SnapshotStepSaver(object):
	def __init__(self, every = 500, base_name = 'net'):
		self.every = every
		self.base_path = join(NNET_STATE_DIR, base_name)

	def __call__(self, nn, train_history):
		epoch = train_history[-1]['epoch']
		if epoch % self.every == 0:
			filepath = '{0:s}_{1:d}.net.npz'.format(self.base_path, epoch)
			save_knowledge(nn, filepath)
			if VERBOSITY >= 1:
				print 'saved network to "{0:s}" at iteration {1:d}'.format(filepath, epoch)


#class WriteOutputLogLoss(_ProbaScorer):
#	def __call__(self, clf, X, y, sample_weight = None):
#		y_pred = clf.predict_proba(X)
#       # plot stuff
#		return log_loss(y, y_pred)


class SnapshotEndSaver(object):
	def __init__(self, base_name = 'net_done'):
		self.base_path = join(NNET_STATE_DIR, base_name)

	def __call__(self, nn, train_history):
		filepath = '{0:s}_complete.net.npz'.format(self.base_path)
		save_knowledge(nn, filepath)
		if VERBOSITY >= 1:
			print 'saved network to "{0:s}" after training ended'.format(filepath)


class TrainProgressPlotter(object):

	def __init__(self, base_name = 'net_hist'):
		self.base_path = join(AUTO_IMAGES_DIR, base_name) + '.png'

	def __call__(self, nn, train_history):
		train = [d['train_loss'] for d in train_history]
		valid = [d['valid_loss'] for d in train_history]
		if len(train) >= 3:
			#import matplotlib
			#matplotlib.use('Agg')
			from matplotlib.pyplot import subplots
			fig, ax = subplots(figsize = (6, 4))
			ax.plot(train, color = 'blue', label = 'train')
			ax.plot(valid, color = 'red', label = 'test')
			ax.legend()
			ax.set_xlim([0, max(10, len(train))])
			ax.set_ylim([0, 1.05 * max(max(train), max(valid))])
			fig.savefig(self.base_path)


