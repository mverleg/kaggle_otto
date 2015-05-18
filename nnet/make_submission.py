
from genericpath import isfile
from nnet.nnio import load_knowledge, save_knowledge
from os.path import join
from nnet.make_net import make_net
from nnet.prepare import conormalize_data
from nnet.train_test import make_pretrain
from nnet.visualization import show_train_progress
from settings import SUBMISSIONS_DIR, PRETRAIN_DIR, NNET_STATE_DIR
from utils.features import chain_feature_generators
from utils.ioutil import makeSubmission as make_submission
from utils.loading import get_training_data, get_testing_data
from matplotlib.pyplot import show
from utils.postprocess import scale_to_priors
from utils.shuffling import shuffle
from numpy import bincount, float64
from validation.score import calc_logloss


print '>> loading data'
train, labels, features = get_training_data()
test = get_testing_data()[0]

print '>> adding test data to train'
print '...skipped'
# train, labels = exand_from_test(train, labels, get_testing_data()[0], confidence = 0.95)

print '>> generating features'
extra_feature_count = None
train, test = chain_feature_generators(train, labels, test, extra_features = extra_feature_count, multiplicity = 3)

print '>> normalizing data'
train, test = conormalize_data(train, test, use_log = True)

print '>> shuffling data'
train, labels, key = shuffle(train, labels)

print '>> removing outliers'
print '...skipped'
# train, labels = filter_data(train, labels, cut_outlier_frac = 0.06, method = 'OCSVM')

print '>> setting parameters'
params = {
	'name': 'single',
	'dense1_nonlinearity': 'rectify',   # tanh, sigmoid, rectify, leaky2, leaky20, softmax
	'dense1_init': 'glorot_normal',     # orthogonal, sparse, glorot_normal, glorot_uniform, he_normal, he_uniform
	'dense1_size': 300,                 # hidden neurons in layer (30~1000)
	'dense2_size': 0,
	'dense3_size': None,
	'learning_rate': 0.0007,            # initial learning rate (learning rate is effectively higher for higher momentum)
	'learning_rate_scaling': 100,       # progression over time; 0.1 scaled by 10 is 0.01
	'momentum': 0.9,                    # initial momentum
	'momentum_scaling': 10 ,            # 0.9 scaled by 10 is 0.99
	'dropout1_rate': 0.5,               # [0, 0.5]
	'dropout2_rate': None,              # inherit dropout1_rate if dense2 exists
	'dropout3_rate': None,              # inherit dropout2_rate if dense3 exist
	'weight_decay': 0.00007,            # constrain the weights to avoid overfitting
	'max_epochs': 600,                  # it terminates when overfitting or increasing, so just leave high
	'auto_stopping': True,              # stop training automatically if it seems to be failing
	'save_snapshots_stepsize': None,    # save snapshot of the network every X epochs
}

print '>> making network'
net = make_net(train.shape[1], **params)

pretrain = join(PRETRAIN_DIR, '{0:s}_pretrain_{1:d}_{2:d}_{3:d}.net.npz'.format(params['name'], params['dense1_size'] or 0, params['dense2_size'] or 0, params['dense3_size']or 0))
if not isfile(pretrain):
	print '>> pretraining network'
	make_pretrain(pretrain, train, labels, extra_feature_count = extra_feature_count, **params)

print '>> loading pretrained network'
load_knowledge(net, pretrain)

print '>> training network'
out = net.fit(train, labels - 1)

print '>> saving network'
save_knowledge(net, join(NNET_STATE_DIR, 'single_trained.net.npz'))

print '>> calculating train error'
prediction = net.predict_proba(train)
prediction = scale_to_priors(prediction, priors = bincount(labels)[1:] / float64(len(labels)))
print 'train loss: {0:.4f} / {0:.4f} (unscaled / scaled)'.format(calc_logloss(prediction, labels))

print '>> predicting test data'
prediction = net.predict_proba(test)

print '>> scaling to priors'
prediction = scale_to_priors(prediction, priors = bincount(labels)[1:] / float64(len(labels)))

print '>> making submission file'
make_submission(prediction, fname = join(SUBMISSIONS_DIR, 'single.csv'), digits = 8)

print '>> plotting training progress'
fig, ax = show_train_progress(net)

print '>> done!'

if __name__ == '__main__':
	show()


