
from numpy import load, save
from nnet.base_optimize import name_from_file
from nnet.train_test import train_test_NN
from utils.loading import get_training_data
from utils.shuffling import shuffle
from validation.score import calc_logloss


test = load('/home/mark/testmat.npy')
train = load('/home/mark/trainmat.npy')
train_labels = load('/home/mark/trainclas.npy')
test_labels = load('/home/mark/testclas.npy')
train, train_labels, key = shuffle(train, train_labels)
test, test_labels, key = shuffle(test, test_labels)

# for some reason, it works on the loaded data but not on the data you sent me
all_data, all_labels = get_training_data()[:2]
k = int(0.9 * all_data.shape[0])
train = all_data[:k, :]
train_labels = all_labels[:k]
test = all_data[k:, :]
test_labels = all_labels[k:]


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


