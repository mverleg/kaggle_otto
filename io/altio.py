
"""
	Load train and test data.
"""

from simplejson import JSONDecoder, dumps
from tempfile import gettempdir
from numpy import array, uint16, load, save
from collections import OrderedDict
from os.path import join


def load_training_data(filepath):
	"""
		Load training data.

		:return: data[ndarray], samples:classes[OrderedDict], features[list]

		Notes:
		* I tried using Pandas to have a dataframe with labels, but it seems I can't skip class label column. -Mark
		* The samples are numbered from 1 because the data file does that. Sorry.
	"""
	with open(filepath, 'r') as fh:
		cells = [line.split(',') for line in fh.read().splitlines()]
	featureli = cells[0][1:-1]
	classes = OrderedDict()
	datali = []
	for row in cells[1:]:
		classes[int(row[0])] = uint16(row[-1].split('_')[-1])
		datali.append(list(uint16(val) for val in row[1:-1]))
	data = array(datali)
	features = array(featureli)
	return data, classes, features


def get_training_data(filepath = 'data/train.csv'):
	"""
		Gets the training data from the CSV file, caching it in temporary directory for speed.
	"""
	try:
		data = load(join(gettempdir(), 'cache_data.npy'))
		features = load(join(gettempdir(), 'cache_features.npy'))
		with open(join(gettempdir(), 'cache_features.npy'), 'r') as fh:
			classes = JSONDecoder(object_pairs_hook = OrderedDict).decode(fh.read())
		print 'got data from cache in {0:s}'.format(tempdir)
	except:
		data, classes, features = load_training_data(filepath = filepath)
		save(join(gettempdir(), 'cache_data.npy'), data)
		save(join(gettempdir(), 'cache_features.npy'), features)
		with open(join(gettempdir(), 'cache_features.npy'), 'w+') as fh:
			fh.write(dumps(features))
		print 'loaded the data directly'
	return data, classes, features


if __name__ == '__main__':
	get_training_data()


