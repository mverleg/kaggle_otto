from numpy import load, save, uint16

from nnet.oldstyle.base_optimize import name_from_file
from nnet.train_test import train_test_NN
from utils.shuffling import shuffle
from validation.score import calc_logloss


train = load('/home/mark/trainmat.npy').astype(uint16)[:, 1:]
test = load('/home/mark/testmat.npy').astype(uint16)[:, 1:]
train_labels = load('/home/mark/trainclas.npy')
test_labels = load('/home/mark/testclas.npy')
train, train_labels, key = shuffle(train, train_labels)

print train.shape, test.shape, train_labels.shape, test_labels.shape
print train.dtype, test.dtype, train_labels.dtype, test_labels.dtype


# parameters based on tainted cross validation (and speed); quite possibly won't perform well now
params = {
	'dense1_nonlinearity': 'rectify',
	'dense1_init': 'glorot_normal',
	'dense1_size': 256,
	'dense2_size': 0,
	'dense3_size': None,
	'learning_rate': 0.001,
	'learning_rate_scaling': 100,
	'momentum': 0.9,
	'momentum_scaling': 10,
	'dropout1_rate': 0.5,
	'dropout2_rate': None,
	'dropout3_rate': None,
	'weight_decay': 0,
	'max_epochs': 1500,
	'auto_stopping': False,
	'extra_feature_count': 0,
	'pretrain': False,  # keep this OFF
	'save_snapshots_stepsize': 500,
	'name': name_from_file(),
	'outlier_method': 'EE',
	'outlier_frac': None,
	'normalize_log': True,
	'use_calibration': False,
	'use_rescale_priors': True,
	'extra_feature_seed': 0,
	'test_data_confidence': None,
}

prediction = train_test_NN(train, train_labels, test, **params)

# do things with prediction
print calc_logloss(prediction, test_labels)
save('nnpred.npy', prediction)


