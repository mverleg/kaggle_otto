
"""
	http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
"""
from lasagne.nonlinearities import softmax

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from settings import NCLASSES
from utils.loading import get_training_data
from utils.shuffling import shuffle


X, y, features = get_training_data()
print X.shape


net = NeuralNet(
	layers = [
		('input', layers.InputLayer),
		('hidden', layers.DenseLayer),
		('output', layers.DenseLayer),
		],

	input_shape = (None, 93),
	hidden_num_units = 75,
	output_nonlinearity = softmax,  # output layer uses identity function  #todo
	output_num_units = NCLASSES,

	update = nesterov_momentum,
	update_learning_rate = 0.0000001,
	update_momentum = 0.9,

	regression = False,
	max_epochs = 3,
	verbose = 1,
)

print [q for q in dir(net) if not q.startswith('_')]

B, j, key = shuffle(data = X, classes = y)
j -= 1
net.fit(B, j)


