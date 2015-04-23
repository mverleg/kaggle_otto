
"""
	Tutorial from:

	http://nbviewer.ipython.org/github/craffel/Lasagne-tutorial/blob/master/examples/tutorial.ipynb
"""

from lasagne.layers import get_all_params, InputLayer, DenseLayer
from lasagne.nonlinearities import tanh, softmax
from lasagne.updates import sgd
from theano import function, config
import theano.tensor as T
import numpy as np
from sklearn.datasets import make_classification
from matplotlib.pyplot import subplots, show, cm

# Generate synthetic data
N_CLASSES = 4
X, y = make_classification(n_samples = 50, n_features=2, n_redundant=0, n_classes=N_CLASSES, n_clusters_per_class=1)
# Convert to theano floatX
X = X.astype(config.floatX)
# Labels should be ints
y = y.astype('int32')
# Make a scatter plot where color encodes class
fig1, ax1 = subplots()
ax1.scatter(X[:, 0], X[:, 1], c=y)


# First, construct an input layer.
# The shape parameter defines the expected input shape, which is just the shape of our data matrix X.
l_in = InputLayer(shape=X.shape)
# A dense layer implements a linear mix (xW + b) followed by a nonlinearity.
l_hidden = DenseLayer(
	l_in,  # The first argument is the input to this layer
	num_units=20,  # This defines the layer's output dimensionality
	nonlinearity=tanh)  # Various nonlinearities are available
l_hidden2 = DenseLayer(l_hidden, num_units=15, nonlinearity=tanh)
# For our output layer, we'll use a dense layer with a softmax nonlinearity.
l_output = DenseLayer(l_hidden2, num_units=N_CLASSES, nonlinearity=softmax)
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
updates = sgd(loss, all_params, learning_rate = 1)
# Finally, we can compile Theano functions for training and computing the output.
train = function([net_input, true_output], loss, updates=updates)
get_output = function([net_input], net_output)


# Train for 100 epochs
for k, n in enumerate(xrange(100)):
	print train(X, y) # this is logloss


# Compute the predicted label of the training data.
# The argmax converts the class probability output to class label
probabilities = get_output(X)  # normalized
y_predicted = np.argmax(probabilities, axis=1)
print (y_predicted == y).mean()
# Plot incorrectly classified points as black dots
fig2, ax21 = subplots()
ax21.scatter(X[:, 0], X[:, 1], c=(y != y_predicted), cmap=cm.gray_r)
# Compute and display the accuracy
ax21.set_title("Accuracy: {0:.2f}%".format(100*np.mean(y == y_predicted)))


#todo: after Other useful functionality
#todo: especially dropout
#todo: extract good weights and use for next training


if __name__ == '__main__':
	show()


