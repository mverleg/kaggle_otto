
from numpy import inf
from os.path import join
from nnet.nnio import save_net, save_knowledge
from settings import NNET_STATE_DIR


class StopWhenOverfitting(object):
	def __init__(self, loss_fraction, base_name):
		self.loss_fraction = loss_fraction
		self.base_path = join(NNET_STATE_DIR, base_name)

	def __call__(self, nn, train_history):
		if train_history[-1]['train_loss'] / train_history[-1]['valid_loss'] <= self.loss_fraction:
			print 'Terminating training since the network is starting to overfit too much.'
			filepath = '{0:s}_{1:d}.net.npz'.format(self.base_path, train_history[-1]['epoch'])
			save_knowledge(nn, filepath)
			raise StopIteration('overfitting')


class StopAfterMinimum(object):
	"""
		From http://danielnouri.org/notes/category/deep-learning/
	"""
	def __init__(self, patience = 70, base_name = 'net'):
		self.patience = patience
		self.best_valid = inf
		self.best_valid_epoch = 0
		self.best_weights = None
		self.base_path = join(NNET_STATE_DIR, base_name)

	def __call__(self, nn, train_history):
		current_valid = train_history[-1]['valid_loss']
		current_epoch = train_history[-1]['epoch']
		if current_valid < self.best_valid:
			self.best_valid = current_valid
			self.best_valid_epoch = current_epoch
			self.best_weights = [w.get_value() for w in nn.get_all_params()]
		elif self.best_valid_epoch + self.patience < current_epoch:
			print 'Stopping early since test error has been increasing.'
			print 'Best validation loss was {:.6f} at epoch {}.'.format(self.best_valid, self.best_valid_epoch)
			filepath = '{0:s}_{1:d}.net.npz'.format(self.base_path, train_history[-1]['epoch'])
			save_knowledge(nn, filepath)
			nn.load_weights_from(self.best_weights)
			filepath = '{0:s}_{1:d}_best.net.npz'.format(self.base_path, self.best_valid_epoch)
			save_knowledge(nn, filepath)
			print 'The network has been restored to the state at this epoch and both have been saved.'
			raise StopIteration('loss increasing')


