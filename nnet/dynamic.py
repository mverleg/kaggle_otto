
"""
	http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
"""

from numpy import linspace, logspace, float32, log2


class DynamicVariable(object):
	def __init__(self, name, start, stop):
		self.name = name
		self.start, self.stop = start, stop
		self.space = None


class LinearVariable(DynamicVariable):
	def __call__(self, nn, train_history):
		if self.space is None:
			self.space = linspace(start = self.start, stop = self.stop, num = nn.max_epochs)
		epoch = train_history[-1]['epoch']
		new_value = float32(self.space[epoch - 1])
		getattr(nn, self.name).set_value(new_value)


class LogarithmicVariable(DynamicVariable):
	def __call__(self, nn, train_history):
		if self.space is None:
			self.space = logspace(start = log2(self.start), stop = log2(self.stop), num = nn.max_epochs, base = 2.)
		epoch = train_history[-1]['epoch']
		new_value = float32(self.space[epoch - 1])
		getattr(nn, self.name).set_value(new_value)


