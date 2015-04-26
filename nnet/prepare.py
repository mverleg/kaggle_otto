
"""
	Prepare data for neural network use.
"""

from matplotlib.pyplot import subplots, show
from numpy import log10
from utils.loading import get_training_data


def prepare_data():
	train_data, classes, features = get_training_data()
	train_data = log10(1 + train_data)
	return train_data, classes, features


if __name__ == '__main__':
	train_data, classes, features = prepare_data()
	print 'min  ', train_data.min()
	print 'max  ', train_data.max()
	print 'std  ', train_data.std()
	print 'std>0', train_data[train_data > 0].std()
	fig, ax = subplots()
	ax.hist(train_data.flat)
	show()


