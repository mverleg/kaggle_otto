
"""
	From: https://github.com/dnouri/nolearn/issues/62
"""

from lasagne import regularization
from lasagne.objectives import Objective
from theano import shared


class WeightDecayObjective(Objective):

	def __init__(self, input_layer, decay, **kwargs):
		super(WeightDecayObjective, self).__init__(input_layer = input_layer, **kwargs)
		self.decay = shared(float(decay))
		if decay:  #todo: is this what happens?
			print 'note that setting weight decay ({0:f}) increases the loss as displayed by the training output (different objective); rely on cross validation'

	def get_loss(self, input = None, target = None, deterministic = False, **kwargs):
		loss = super(WeightDecayObjective, self).get_loss(input = input, target = target, deterministic = deterministic, **kwargs)
		if not deterministic:
			return loss + self.decay * regularization.l2(self.input_layer)
		else:
			return loss


