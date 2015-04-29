
from random import random
from nnet.hue.class_split import class_split
from nnet.hue.mahal_outliers import remove_outliers_mahalanobis
from nnet.hue.split_data import split_data
from settings import TRAIN_DATA_PATH
from utils.loading import get_training_data
from utils.shuffling import shuffle


def preprocess(filepath = TRAIN_DATA_PATH, test_frac = 0.1):
	print 'loading data'
	#all_data, all_labels = load_train_data_csv()
	all_data, all_labels = get_training_data(filepath = filepath)[:2]

	print 'shuffling data'
	# seed = random() makes sure it's different each time
	all_data, all_labels = shuffle(all_data, all_labels, seed = random())[:2]

	print 'splitting data into train/test'
	train, train_labels, test, test_labels = split_data(all_data, all_labels, test_frac = test_frac)
	#print train.shape, test.shape, all_data.shape

	print 'splitting into classes'
	class_set, class_set_labels = class_split(train, train_labels)

	print 'outliers using mahalanobis percentage'
	class_set_1, class_set_labels_1, outlier_data_1, outlier_labels_1, smallest_size_1 = remove_outliers_mahalanobis(class_set, class_set_labels, offset = 0.95)
	print 'outliers using mahalanobis treshold'
	class_set_2, class_set_labels_2, outlier_data_2, outlier_labels_2, smallest_size_2 = remove_outliers_mahalanobis(class_set, class_set_labels, offset = 155)

	return class_set_1, class_set_labels_1, outlier_data_1, outlier_labels_1, smallest_size_1, class_set_2, class_set_labels_2, outlier_data_2, outlier_labels_2, smallest_size_2, test, test_labels


