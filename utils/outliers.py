
from settings import TRAIN_DATA_PATH, NCLASSES
from utils.loading import get_training_data


def make_filtered_data(cut_outlier_frac, filepath):
	train_data, true_classes, features = get_training_data()
	for cls in range(1, NCLASSES + 1):
		train_cls = train_data[true_classes == cls, :]

	return train_data, true_classes, features


def get_filtered_data(cut_outlier_frac = 0.05, filepath = TRAIN_DATA_PATH):
	return make_filtered_data(cut_outlier_frac = cut_outlier_frac, filepath = filepath)


# http://scikit-learn.org/stable/modules/outlier_detection.html


