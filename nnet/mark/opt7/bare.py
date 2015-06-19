
from nolearn.lasagne import NeuralNet, BatchIterator
from numpy import float32, log2
from theano import shared
from settings import NCLASSES
from settings import VERBOSITY
from utils.loading import get_preproc_data
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.objectives import Objective
from lasagne.updates import nesterov_momentum
from lasagne.init import HeUniform, Constant
from lasagne.nonlinearities import LeakyRectify, softmax


# https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14303/0-416-in-just-a-few-lines-of-code
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py
# http://lasagne.readthedocs.org/en/latest/modules/updates.html#lasagne.updates.norm_constraint
# norm_constraint

train, labels, test = get_preproc_data(None, expand_confidence = None)
train = log2(1 + train.astype(float32))
train = 3 * train / train.max(0)
labels -= labels.min()

net = NeuralNet(
	layers = [
		('input', InputLayer),
		('dropout0', DropoutLayer),
		('dense1', DenseLayer),
		('dropout1', DropoutLayer),
		('dense2', DenseLayer),
		('dropout2', DropoutLayer),
		('dense3', DenseLayer),
		('dropout3', DropoutLayer),
		('output', DenseLayer),
	],
	update = nesterov_momentum,  #Todo: optimize
	loss = None,
	objective = Objective,
	regression = False,
	max_epochs = 1000,
	eval_size = 0.1,
	#on_epoch_finished = None,
	#on_training_finished = None,
	verbose = bool(VERBOSITY),
	input_shape = (None, train.shape[1]),
	output_num_units = NCLASSES,
	dense1_num_units = 500,
	dense2_num_units = 500,
	dense3_num_units = 400,
	dense1_nonlinearity = LeakyRectify(leakiness = 0.1),
	dense2_nonlinearity = LeakyRectify(leakiness = 0.1),
	dense3_nonlinearity = LeakyRectify(leakiness = 0.1),
	output_nonlinearity = softmax,
	dense1_W = HeUniform(),
	dense2_W = HeUniform(),
	dense3_W = HeUniform(),
	dense1_b = Constant(0.),
	dense2_b = Constant(0.),
	dense3_b = Constant(0.),
	output_b = Constant(0.),
	dropout0_p = 0.1,
	dropout1_p = 0.6,
	dropout2_p = 0.6,
	dropout3_p = 0.6,
	update_learning_rate = shared(float32(0.02)), #
	update_momentum = shared(float32(0.9)), #
	batch_iterator_train = BatchIterator(batch_size = 128),
	batch_iterator_test = BatchIterator(batch_size = 128),
)
net.initialize()


net.fit(train, labels)


