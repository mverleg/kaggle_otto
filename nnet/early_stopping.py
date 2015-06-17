
from sys import stderr
from numpy import inf, isnan, copy
from os.path import join
from nnet.nnio import save_knowledge
from settings import NNET_STATE_DIR, VERBOSITY, DivergenceError


class StopWhenOverfitting(object):
	def __init__(self, loss_fraction, base_name):
		self.loss_fraction = loss_fraction
		self.base_path = join(NNET_STATE_DIR, base_name)

	def __call__(self, nn, train_history):
		if train_history[-1]['epoch'] > 30 and train_history[-1]['train_loss'] / train_history[-1]['valid_loss'] <= self.loss_fraction:
			print 'Terminating training since the network is starting to overfit too much.'
			filepath = '{0:s}_{1:d}.net.npz'.format(self.base_path, train_history[-1]['epoch'])
			if not hasattr(self, 'parent'):
				print 'COULD NOT SAVE NETWORK SINCE {0:s} HAS NO PARENT'.format(self)
			if hasattr(nn, 'parent'):
				nn.parent.save(filepath = filepath)
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
			self.best_weights = copy(nn.get_all_params_values())
		elif self.best_valid_epoch + self.patience < current_epoch:
			print 'Stopping early since test error has been increasing.'
			print 'Best validation loss was {:.6f} at epoch {}.'.format(self.best_valid, self.best_valid_epoch)
			filepath = '{0:s}_{1:d}.net.npz'.format(self.base_path, train_history[-1]['epoch'])
			save_knowledge(nn, filepath)
			nn.load_params_from(self.best_weights)
			filepath = '{0:s}_{1:d}_best.net.npz'.format(self.base_path, self.best_valid_epoch)
			self.parent.save(filepath = filepath)
			print 'The network has been restored to the state at this epoch and both have been saved.'
			raise StopIteration('loss increasing')


class StopNaN(object):
	def __init__(self, raise_divergence = True):
		self.raise_divergence = raise_divergence

	def __call__(self, nn, train_history):
		if isnan(train_history[-1]['train_loss']) or isnan(train_history[-1]['valid_loss']):
			stderr.write('STOPPED SINCE LOSS DIVERGED (NaN)\nnetwork will be re-initialized to not crash the cross validator\none possible reason is a zero-column in the data\n')
			if self.raise_divergence:
				raise DivergenceError('Stopped since loss diverged (NaN)')
			nn.initialize()
			raise StopIteration('diverged')


class BreakEveryN():
	def __init__(self, interrupt_step):
		self.interrupt_step = interrupt_step

	def __call__(self, nn, train_history):
		nn._train_history = train_history
		epoch = train_history[-1]['epoch']
		if epoch % self.interrupt_step == 0:
			if VERBOSITY >= 2:
				print 'stopping at {0:d} (every {1:d} steps)'.format(epoch, self.interrupt_step)
			raise StopIteration('stop every {0:d}'.format(self.interrupt_step))


