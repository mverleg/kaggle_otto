
from argparse import ArgumentParser
from sys import argv
from os.path import dirname, realpath, join


# You can add settings here rather than hard-coding them.

SEED = 4242


"""
	Data information.
"""
TRAINSIZE = 61878
TESTSIZE = 144368
NFEATS = 93
NCLASSES = 9


"""
	Special files.
"""
BASE_DIR = dirname(realpath(__file__))
TRAIN_DATA_PATH = join(BASE_DIR, 'data/train.csv')
TEST_DATA_PATH = join(BASE_DIR, 'data/test.csv')
COLUMN_MAX_PATH = join(BASE_DIR, 'data/max.npy')
OPTIMIZE_RESULTS_DIR = join(BASE_DIR, 'results/optimize')
NNET_STATE_DIR = join(BASE_DIR, 'results/nnets')
AUTO_IMAGES_DIR = join(BASE_DIR, 'results/images')


"""
	Increase verbosity like -v or -vv.
"""
parser = ArgumentParser()
parser.add_argument('-v', action = 'count', default = 0, dest = 'verbosity', help = 'Add up to three -v args to increase verbosity.')
args = parser.parse_known_args(argv[1:])[0]
VERBOSITY = args.verbosity


