
from nnet.nnio import load_knowledge
from os.path import join
from nnet.make_net import make_net
from nnet.prepare import normalize_data, conormalize_data
from nnet.visualization import show_train_progress
from settings import SUBMISSIONS_DIR
from utils.features import chain_feature_generators
from utils.ioutil import makeSubmission as make_submission
from utils.loading import get_training_data, get_testing_data
from utils.outliers import filter_data
from matplotlib.pyplot import show
from utils.shuffling import shuffle


print '>> loading data'
train, labels, features = get_training_data()
test = get_testing_data()[0]

print '>> generating features'
train, test = chain_feature_generators(train, labels, test, extra_features = 63, multiplicity = 3)

print '>> normalizing data'
train, test = conormalize_data(train, test, use_log = True)

print '>> shuffling data'
train, labels, key = shuffle(train, labels)

print '>> removing outliers'
print '...skipped'
#train, labels = filter_data(train, labels, cut_outlier_frac = 0.06, method = 'OCSVM')

print '>> setting parameters'
params = {
	'name': 'single',                   # just choose something sensible
	'dense1_nonlinearity': 'leaky20',   # tanh, sigmoid, rectify, leaky2, leaky20, softmax
	'dense1_init': 'glorot_uniform',    # orthogonal, sparse, glorot_normal, glorot_uniform, he_normal, he_uniform
	'dense1_size': 512,                 # hidden neurons in layer (30~1000)
	'dense2_size': 512,
	'dense3_size': 512,
	'learning_rate': 0.001,             # initial learning rate (learning rate is effectively higher for higher momentum)
	'learning_rate_scaling': 1000,      # pogression over time; 0.1 scaled by 10 is 0.01
	'momentum': 0.9,                    # initial momentum
	'momentum_scaling': 100,            # 0.9 scaled by 10 is 0.99
	'dropout1_rate': 0,                 # [0, 0.5]
	'dropout2_rate': 0,
	'weight_decay': 0,                  # constrain the weights to avoid overfitting
	'max_epochs': 1000,                 # it terminates when overfitting or increasing, so just leave high
	'auto_stopping': True,              # stop training automatically if it seems to be failing
	'verbosity': True,
}

print '>> making network'
net = make_net(train.shape[1], **params)

#print '>> loading pretrained network'
#load_knowledge(net, 'results/nnets/single_pretrain.net.npz')

print '>> training network'
out = net.fit(train, labels - 1)

print '>> predicting test data'
prediction = net.predict_proba(test)

print '>> making submission file'
make_submission(prediction, fname = join(SUBMISSIONS_DIR, 'single.csv'), digits = 8)

print '>> plotting training progress'
fig, ax = show_train_progress(net)

print '>> done!'

if __name__ == '__main__':
	show()


