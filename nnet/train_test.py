
from nnet.nnio import save_knowledge
from os.path import isfile
from nnet.make_net import make_net
from nnet.prepare import normalize_data
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import ShuffleSplit
from utils.outliers import filter_data
from utils.postprocess import scale_to_priors
from numpy import bincount, float64
from validation.score import calc_logloss


def train_NN(train, labels, outlier_frac = 0, outlier_method = 'OCSVM', normalize_log = True,
		use_calibration = False, use_rescale_priors = False, **parameters):
	if outlier_frac:
		train, labels = filter_data(train, labels, cut_outlier_frac = outlier_frac, method = outlier_method)
	train, norms = normalize_data(train, use_log = normalize_log)  # also converts to floats
	net = make_net(**parameters)
	if use_calibration:
		net = CalibratedClassifierCV(net, method = 'sigmoid', cv = ShuffleSplit(train.shape[0], n_iter = 1, test_size = 0.15))
	net.fit(train, labels - 1)
	return net


def train_test_NN(train, labels, test, use_rescale_priors = False, **parameters):
	net = train_NN(train, labels, outlier_frac = 0, outlier_method = 'OCSVM', normalize_log = True,
			use_calibration = False, use_rescale_priors = False, **parameters)
	prediction = net.predict_proba(test)
	if use_rescale_priors:
		prediction = scale_to_priors(prediction, priors = bincount(labels)[1:] / float64(len(labels)))
	return prediction


def make_pretrain(pretrain_path, data, labels, minimum_train_loss = 0.7, **params):
	if not isfile(pretrain_path):
		print 'pretraining file not found, pretraining a network now'
		pretrain_params = dict(params.items() + [('pretrain', None)])
		net = train_NN(data, labels, **pretrain_params)
		assert calc_logloss(net.predict_proba(data), labels) < minimum_train_loss, 'Pre-training did not converge'
		save_knowledge(net, pretrain_path)


