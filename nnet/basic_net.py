
"""
	Based on playground/tutorial1.py
"""

from lasagne.layers import get_all_params, InputLayer, DenseLayer
from lasagne.nonlinearities import tanh, softmax
from lasagne.updates import sgd
from theano import function, config
import theano.tensor as T
import numpy as np
from sklearn.datasets import make_classification
from matplotlib.pyplot import subplots, show, cm
from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator


train_data, true_classes, features = get_training_data()
validator = SampleCrossValidator(train_data, true_classes, test_frac = 0.3)
for train, classes, test in validator.yield_cross_validation_sets(rounds = 13):
	# create tensor objects
	trainT = train.astype(config.floatX)
	testT = train.astype(config.floatX)
	classT = classes.astype('int32')


	# First, construct an input layer.
	# The shape parameter defines the expected input shape, which is just the shape of our data matrix X.
	l_in = InputLayer(shape = trainT.shape)
	# A dense layer implements a linear mix (xW + b) followed by a nonlinearity.
	l_hidden = DenseLayer(
		l_in,  # The first argument is the input to this layer
		num_units = 25,  # This defines the layer's output dimensionality
		nonlinearity = tanh)  # Various nonlinearities are available
	# For our output layer, we'll use a dense layer with a softmax nonlinearity.
	l_output = DenseLayer(l_hidden, num_units = len(classes), nonlinearity = softmax)
	# Now, we can generate the symbolic expression of the network's output given an input variable.
	net_input = T.matrix('net_input')
	net_output = l_output.get_output(net_input)

	# As a loss function, we'll use Theano's categorical_crossentropy function.
	# This allows for the network output to be class probabilities,
	# but the target output to be class labels.
	true_output = T.ivector('true_output')
	loss = T.mean(T.nnet.categorical_crossentropy(net_output, true_output))
	# Retrieving all parameters of the network is done using get_all_params,
	# which recursively collects the parameters of all layers connected to the provided layer.
	all_params = get_all_params(l_output)
	# Now, we'll generate updates using Lasagne's SGD function
	updates = sgd(loss, all_params, learning_rate = 0.01)
	# Finally, we can compile Theano functions for training and computing the output.
	training = function([net_input, true_output], loss, updates=updates)
	prediction = function([net_input], net_output)

	# Train for 100 epochs
	print 'epoch  logloss'
	for k, n in enumerate(xrange(100)):
		# this is logloss
		print '{0:.3d}  {1:.4f}'.format(k, training(trainT, classT))

	# Compute the predicted label of the training data.
	# The argmax converts the class probability output to class label
	probabilities = prediction(testT)  # normalized
	prediction = np.argmax(probabilities, axis=1)

	# cross validation
	validator.add_prediction(prediction)

validator.print_results()


if __name__ == '__main__':
	show()


