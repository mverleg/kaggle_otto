
from nnet.nnio import save_knowledge
from os.path import isfile
from nnet.make_net import make_net
from nnet.prepare import conormalize_data, normalize_data
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import ShuffleSplit
from utils.features import PositiveSparseFeatureGenerator
from utils.outliers import filter_data
from utils.postprocess import scale_to_priors
from numpy import bincount, float64
from validation.score import calc_logloss


def train_NN(train, labels, outlier_frac = 0, outlier_method = 'OCSVM', use_calibration = False,
		use_rescale_priors = False, normalize_log = True, extra_feature_count = 57,
		extra_feature_seed = 0, **parameters):
	if outlier_frac:
		train, labels = filter_data(train, labels, cut_outlier_frac = outlier_frac, method = outlier_method)
	net = make_net(**parameters)
	if use_calibration:
		net = CalibratedClassifierCV(net, method = 'sigmoid', cv = ShuffleSplit(train.shape[0], n_iter = 1, test_size = 0.15))
	net.fit(train, labels - 1)
	return net


def train_test_NN(train, labels, test, use_rescale_priors = False, normalize_log = True, extra_feature_count = 57,
		extra_feature_seed = 0, **parameters):
	train, test = conormalize_data(train, test, use_log = normalize_log)
	if extra_feature_count:
		gen = PositiveSparseFeatureGenerator(train, labels, difficult_classes = (2, 3),
			extra_features = extra_feature_count, seed = extra_feature_seed)
		train, test = gen.add_features_tt(train, test)
	net = train_NN(train, labels, use_rescale_priors = use_rescale_priors, **parameters)
	prediction = net.predict_proba(test)
	if use_rescale_priors:
		prediction = scale_to_priors(prediction, priors = bincount(labels)[1:] / float64(len(labels)))
	return prediction


def make_pretrain(pretrain_path, data, labels, minimum_train_loss = 0.7, normalize_log = True, **params):
	if not isfile(pretrain_path):
		print 'pretraining file not found, pretraining a network now'
		pretrain_params = dict(params.items() + [('pretrain', None)])
		data = normalize_data(data, use_log = normalize_log)[0]
		net = train_NN(data, labels, **pretrain_params)
		train_err = calc_logloss(net.predict_proba(data), labels)
		assert train_err < minimum_train_loss, 'Pre-training did not converge ({0:.4f} >= {1:.4f})'.format(train_err, minimum_train_loss)
		save_knowledge(net, pretrain_path)


