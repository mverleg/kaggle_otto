
"""
	http://nbviewer.ipython.org/github/craffel/Lasagne-tutorial/blob/master/examples/tutorial.ipynb
"""

import theano
import theano.tensor as T
import lasagne
import numpy as np
import sklearn.datasets
from matplotlib.pyplot import subplots, show


# Generate synthetic data
N_CLASSES = 4
X, y = sklearn.datasets.make_classification(n_features=2, n_redundant=0,
											n_classes=N_CLASSES, n_clusters_per_class=1)
# Convert to theano floatX
X = X.astype(theano.config.floatX)
# Labels should be ints
y = y.astype('int32')
# Make a scatter plot where color encodes class
plt.scatter(X[:, 0], X[:, 1], c=y)


# First, construct an input layer.
# The shape parameter defines the expected input shape, which is just the shape of our data matrix X.
l_in = lasagne.layers.InputLayer(shape=X.shape)
# A dense layer implements a linear mix (xW + b) followed by a nonlinearity.
l_hidden = lasagne.layers.DenseLayer(
	l_in,  # The first argument is the input to this layer
	num_units=10,  # This defines the layer's output dimensionality
	nonlinearity=lasagne.nonlinearities.tanh)  # Various nonlinearities are available
# For our output layer, we'll use a dense layer with a softmax nonlinearity.
l_output = lasagne.layers.DenseLayer(l_hidden, num_units=N_CLASSES,
									 nonlinearity=lasagne.nonlinearities.softmax)
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
all_params = lasagne.layers.get_all_params(l_output)
# Now, we'll generate updates using Lasagne's SGD function
updates = lasagne.updates.sgd(loss, all_params, learning_rate=1)
# Finally, we can compile Theano functions for training and computing the output.
train = theano.function([net_input, true_output], loss, updates=updates)
get_output = theano.function([net_input], net_output)


# Train for 100 epochs
for n in xrange(100):
	train(X, y)


# Compute the predicted label of the training data.
# The argmax converts the class probability output to class label
y_predicted = np.argmax(get_output(X), axis=1)
# Plot incorrectly classified points as black dots
plt.scatter(X[:, 0], X[:, 1], c=(y != y_predicted), cmap=plt.cm.gray_r)
# Compute and display the accuracy
plt.title("Accuracy: {}%".format(100*np.mean(y == y_predicted)))


#todo: after Other useful functionality
#todo: especially dropout



if __name__ == '__main__':
	show()


