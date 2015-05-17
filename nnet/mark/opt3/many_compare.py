
from multiprocessing import Pool
from random import Random
from os.path import isfile, join
from nnet.base_optimize import optimize_NN, name_from_file
from settings import SEED, NNET_STATE_DIR


def optimize_map(index):
	random = Random(index + SEED)
	name = 'many_{1:d}'.format(name_from_file(), index)
	if isfile(join(NNET_STATE_DIR, name + '_best.net.npz')):
		print name, 'best found'
		pretrain = join(NNET_STATE_DIR, name + '_best.net.npz')
	elif isfile(join(NNET_STATE_DIR, name + '.net.npz')):
		print name, 'found'
		pretrain = join(NNET_STATE_DIR, name + '.net.npz')
	else:
		print name, 'NOT found'
		pretrain = None
		#return
	params = {
		'name': name,
		'dense1_size': 64 + random.randint(0, 384),
		'dense2_size': 64 + random.randint(0, 384),
		'dense3_size': 64 + random.randint(0, 128),
		'learning_rate': 0.001 + 0.01 * random.random(),
		'learning_rate_scaling': 500,
		'momentum': 0.99,
		'momentum_scaling': 10,
		'dropout1_rate': 0.5,
		'dropout2_rate': 0.5,
		'weight_decay': 0.00025 * (random.random() + random.random() + random.random()),
		'max_epochs': 1500,
		'extra_feature_count': 163,
		'pretrain': pretrain,
		'save_snapshots_stepsize': 1500,
	}
	if index < 10:
		print index, params
	return
	return optimize_NN(debug = True, test_only = True, **params)

N = 600
pool = Pool(processes = 1)
pool.map(optimize_map, range(N))


