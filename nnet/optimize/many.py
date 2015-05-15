
from json import dump
from multiprocessing import Pool
from os.path import join
from random import Random
from nnet.base_optimize import optimize_NN, name_from_file
from settings import SEED, NNET_STATE_DIR


def optimize_map(index):
	random = Random(index + SEED)
	params = {
		'name': '{0:s}_{1:d}'.format(name_from_file(), index),
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
		'pretrain': False,
		'save_snapshots_stepsize': 1500,
	}
	with open(join(NNET_STATE_DIR, 'param_{0:03d}.json'.format(index)), 'w+') as fh:
		print 'writing params for #{0:d}'.format(index)
		dump(obj = params, fp = fh)
	return optimize_NN(debug = True, **params)

N = 600
pool = Pool(processes = min(N, 60))
pool.map(optimize_map, range(N))


