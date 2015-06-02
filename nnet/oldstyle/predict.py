
"""
	Make a prediction with an existing network.
"""

from copy import copy
from numpy import load
from nnet.nnio import load_knowledge
from nnet.oldstyle.train_test import train_NN
from utils.loading import get_training_data
from utils.normalize import normalized_sum
from utils.postprocess import scale_to_priors
from validation.score import calc_logloss


def predict(parameters, networkfile, data):
	"""
		Calculate probabilities for the data.

		:param parameters: parameters for the network (must match networkfile)
		:param networkfile: .net.npz file for the network
		:param data: ndarray with data (train or test)
		:return: probabilities

		Doesn't work with added test data or with outlier removal.
	"""
	parameters = copy(parameters)
	parameters.update({'verbosity': False, 'pretrain': networkfile})
	net = train_NN(data, labels = None, test = None, test_only = True, **parameters)[0]
	load_knowledge(net, networkfile)
	prediction = net.predict_proba(data)
	scale_to_priors(prediction, priors = normalized_sum([1929, 16122, 8004, 2691, 2739, 14135, 2839, 8464, 4955]))
	print 'predicted {0:d} samples'.format(prediction.shape[0])
	return prediction


def predict_300_v1():
	data, labels = get_training_data()[:2]
	params = {
		'dense1_nonlinearity': 'rectify',
		'dense1_init': 'glorot_normal',
		'dense1_size': 300,
		'dense2_size': 0,
		'dense3_size': None,
		'dropout1_rate': 0.5,
		'dropout2_rate': None,
		'dropout3_rate': None,
		'extra_feature_count': 0,
	}
	probs = predict(params, 'results/pretrain/single_pretrain_300_0_0.net.npz', data)
	print 'logloss', calc_logloss(probs, labels)


def predict_ensemble():
	test = load('/home/mark/testmat.npy')
	train = load('/home/mark/trainmat.npy')
	train_labels = load('/home/mark/trainclas.npy')
	test_labels = load('/home/mark/testclas.npy')
	params = {
		'dense1_nonlinearity': 'rectify',
		'dense1_init': 'glorot_normal',
		'dense1_size': 300,
		'dense2_size': 0,
		'dense3_size': None,
		'dropout1_rate': 0.5,
		'dropout2_rate': None,
		'dropout3_rate': None,
		'extra_feature_count': 0,
	}
	probs = predict(params, 'results/pretrain/single_pretrain_300_0_0.net.npz', train)
	print 'logloss', calc_logloss(probs, train_labels)


if __name__ == '__main__':
	predict_300_v1()


