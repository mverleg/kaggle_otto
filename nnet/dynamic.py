
"""
	http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
"""

from matplotlib.pyplot import subplots, show
from numpy import linspace, logspace, float32, log2


class DynamicVariable(object):
	def __init__(self, name, start, stop, epoch_count = 2000):
		self.name = name
		self.start, self.stop = start, stop
		self.epoch_count = epoch_count
		self.space = None


class LinearVariable(DynamicVariable):
	def __call__(self, nn, train_history):
		if self.space is None:
			self.space = linspace(start = self.start, stop = self.stop, num = min(nn.max_epochs, self.epoch_count))
			#fig, ax = subplots()  # tmp
			#ax.plot(self.space)
			#ax.set_title(self.name)
			#show()
		epoch = train_history[-1]['epoch']
		if epoch < self.epoch_count:
			new_value = float32(self.space[epoch - 1])
			getattr(nn, self.name).set_value(new_value)


class LogarithmicVariable(DynamicVariable):
	def __call__(self, nn, train_history):
		if self.space is None:
			self.space = logspace(start = log2(self.start), stop = log2(self.stop), num = min(nn.max_epochs, self.epoch_count), base = 2.)
		epoch = train_history[-1]['epoch']
		if epoch < self.epoch_count:
			new_value = float32(self.space[epoch - 1])
			getattr(nn, self.name).set_value(new_value)


if __name__ == '__main__':
	q = logspace(start = log2(0.003), stop = log2(0.003/1000), num = 1000, base = 2.)
	z = linspace(start = 0.003, stop = 0.003/1000, num = 1000)
	fig, ax = subplots(figsize = (6, 3))
	#fig.tight_layout()
	ax.set_xlabel('epochs')
	ax.set_ylabel('learning rate')
	ax.plot(q, c = 'b')
	ax.plot(z, c = 'r')
	ax.grid()
	show()


