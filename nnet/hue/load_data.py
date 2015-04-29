
from numpy import loadtxt


def load_train_data_csv():
	# load the trian and test data without any shuffling / normalziing / filtering / ...
	train_data = loadtxt('nnet/hue/data/train_noheader.csv', delimiter = ',')
	true_labels = loadtxt('nnet/hue/data/labels.csv', delimiter = ',')
	return train_data, true_labels


def load_test_data_csv():
	return loadtxt('nnet/hue/data/test_noheader.csv', delimiter = ',')



