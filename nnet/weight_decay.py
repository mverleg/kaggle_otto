
"""
	From: https://github.com/dnouri/nolearn/issues/62

	Printing: https://groups.google.com/forum/#!topic/theano-users/5yuHeWI2KPM
"""

from numpy import mean
from lasagne import regularization
from lasagne.objectives import Objective
from theano import shared
from settings import VERBOSITY


class WeightDecayObjective(Objective):

	def __init__(self, input_layer, weight_decay_holder, **kwargs):
		super(WeightDecayObjective, self).__init__(input_layer = input_layer, **kwargs)
		self.weight_decay_holder = weight_decay_holder
		if self.weight_decay_holder[0]:  #todo: is this what happens?
			print 'note that setting weight decay ({0:.3g}) increases the loss as displayed by the training output (different objective); rely on cross validation'.format(self.weight_decay_holder[0])

	def get_loss(self, input = None, target = None, deterministic = False, **kwargs):
		loss = super(WeightDecayObjective, self).get_loss(input = input, target = target, deterministic = deterministic, **kwargs)
		if not deterministic:
			return loss + self.weight_decay_holder[0] * regularization.l2(self.input_layer)
		else:
			return loss


class AdaptiveWeightDecay(object):
	"""
		Change the weight decay based on current ratio of train and test data.
	"""
	def __init__(self, increase_factor = 1.5, increase_trigger = 0.90, decrease_factor = 0.8,
			decrease_trigger = 0.97, trigger_history = 7, cooldown_epochs = 10):
		"""
			:param weight_decay_ref: A shared variable storing the weight decay.
			:param increase_factor: By what factor to increase the weight decay when overfitting.
			:param increase_trigger: At which ratio of test and train loss is the network overfitting?
			:param decrease_factor: Similar to increase_factor.
			:param decrease_trigger: Similar to increase_trigger.
			:param trigger_history: The average how many epochs to average when calculating trigger value?
			:param cooldown_epochs: Freeze the weight decay for how many epochs?
		"""
		assert 0 <= cooldown_epochs <= 1000, 'Cooldown should be a positive integer (< 1000)'
		assert 0 <= cooldown_epochs <= 1000, 'History length should be a positive integer (< 1000)'
		assert increase_trigger < decrease_trigger or increase_trigger is None or decrease_trigger is None, 'Increase trigger value should be smaller than decrease one (to prevent both increase and decrease activating)'
		assert increase_factor >= 1, 'increase_factor should increase the weight decay (that is, be larger than 1)'
		assert decrease_factor <= 1, 'decrease_factor should decrease the weight decay (that is, be smaller than 1)'
		self.cooldown = cooldown_epochs
		self.hist = trigger_history
		self.increase_factor = shared(increase_factor)
		self.increase_trigger = increase_trigger
		self.decrease_factor = shared(decrease_factor)
		self.decrease_trigger = decrease_trigger
		self.countdown = 0 #self.cooldown todo cooldown_epochs

	def __call__(self, nn, train_history):
		self.countdown -= 1
		if self.countdown > 0:
			return
		ratio = mean([d['valid_loss'] for d in train_history[-self.hist:]]) / mean([d['train_loss'] for d in train_history[-self.hist:]])
		if ratio < self.increase_trigger:
			if VERBOSITY >= 2:
				print 'increasing weight decay by factor {0:.5f}'.format(self.increase_factor.get_value())
			nn.weight_decay_holder[0] = (nn.weight_decay_holder[0] * self.increase_factor)
			self.countdown = self.cooldown
		elif ratio > self.decrease_trigger:
			if VERBOSITY >= 2:
				print 'decreasing weight decay by factor {0:.5f}'.format(self.decrease_factor.get_value())
			nn.weight_decay_holder[0] = (nn.weight_decay_holder[0] * self.decrease_factor)
			self.countdown = self.cooldown
		else:
			print '(no effect)'


