
"""
	http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
"""

from matplotlib.pyplot import subplots, show
from numpy import array
from lasagne.nonlinearities import softmax
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nnet.prepare import normalize_data
from settings import NCLASSES, VERBOSITY
from utils.shuffling import shuffle


X, y, features = normalize_data()

net = NeuralNet(
	layers = [
		('input', layers.InputLayer),
		('hidden', layers.DenseLayer),
		('output', layers.DenseLayer),
		],

	input_shape = (128, 93),
	hidden_num_units = 25,
	output_nonlinearity = softmax,
	output_num_units = NCLASSES,

	update = nesterov_momentum,
	update_learning_rate = 0.0001,
	update_momentum = 0.9,

	regression = False,
	max_epochs = 500,
	verbose = bool(VERBOSITY),
)

B, j, key = shuffle(data = X, classes = y)
j -= 1
print B.shape
out = net.fit(B, j)

prediction = net.predict(B)

epochs = array([res['epoch'] for res in net.train_history_])
train_loss = array([res['train_loss'] for res in net.train_history_])
valid_loss = array([res['valid_loss'] for res in net.train_history_])

fig, ax = subplots()
ax.plot(epochs, train_loss, color = 'blue', label = 'train')
ax.plot(epochs, valid_loss, color = 'green', label = 'test')
ax.set_xlabel('Log epoch')
ax.set_ylabel('Loss')
ax.set_xscale('log')
ax.legend()


if __name__ == '__main__':
	show()


