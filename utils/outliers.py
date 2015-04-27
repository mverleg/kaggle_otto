
"""
	http://scikit-learn.org/stable/modules/outlier_detection.html
"""

from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from settings import TRAIN_DATA_PATH, NCLASSES, SEED
from utils.loading import get_training_data


def make_filtered_data(cut_outlier_frac, filepath,
		#detector_generator = lambda cut_outlier_frac: OneClassSVM(nu=0.261, gamma=0.05)
		detector_generator = lambda cut_outlier_frac: EllipticEnvelope(contamination = cut_outlier_frac, random_state = SEED)
	):
	assert 0 <= cut_outlier_frac <= 1. + 1e-6
	train_data, true_classes, features = get_training_data()
	for cls in range(1, NCLASSES + 1):
		cls_train = train_data[true_classes == cls, :]
		transformed = PCA(n_components = 40, copy = True, whiten = False).fit(cls_train).transform(cls_train)
		decisions = detector_generator(cut_outlier_frac).fit(transformed).decision_function(transformed)
		print cls, decisions.min(), decisions.max(), decisions.mean(), (decisions < 0).sum(), len(decisions)
	return train_data, true_classes, features


def get_filtered_data(cut_outlier_frac = 0.05, filepath = TRAIN_DATA_PATH):
	return make_filtered_data(cut_outlier_frac = cut_outlier_frac, filepath = filepath)


# http://scikit-learn.org/stable/modules/outlier_detection.html


