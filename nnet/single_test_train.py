
from os.path import join
from nnet.make_net import make_net
from nnet.nnio import load_net
from nnet.prepare import normalize_data
from nnet.visualization import show_train_progress
from settings import SUBMISSIONS_DIR, NNET_STATE_DIR
from utils.ioutil import makeSubmission as make_submission
from utils.loading import get_training_data, get_testing_data
from utils.outliers import filter_data
from matplotlib.pyplot import show
from utils.shuffling import shuffle


print '>> loading train data'
train, classes, features = get_training_data()

print '>> loading test data'
test = get_testing_data()[0]

print '>> normalizing training data'
train, norm = normalize_data(train, use_log = True)

print '>> normalizing testing data'
test = normalize_data(test, norms = norm)[0]

print '>> shuffling data'
train, classes, key = shuffle(train, classes)
# use this to reduce data size in case of memory problems:
# train, classes = train[:1280, :], classes[:1280]

print '>> removing outliers'
train, classes = filter_data(train, classes, cut_outlier_frac = 0.06, method = 'OCSVM')

print '>> making network'
net = make_net(
	name = name,                      # just choose something sensible
	dense1_size = 180,                # [30, 25, 80, 120, 180]
	dense1_nonlinearity = 'tanh',     # ['tanh', 'sigmoid', 'rectify', 'leaky2', 'leaky20' 'softmax']
	dense1_init = 'glorot_uniform',   # ['orthogonal', 'sparse', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
	dense2_size = 180,                # [30, 25, 80, 120, 180]
	dense2_nonlinearity = 'tanh',     # this is coupled to dense1_nonlinearity through hack#1
	dense2_init = 'glorot_uniform',   # idem hack2
	learning_rate = 0.001,            # initial learning reate
	learning_rate_scaling = 10,       # pogression over time; 0.1 scaled by 10 is 0.01
	momentum = 0.99,                  # initial momentum
	momentum_scaling = 10,            # 0.9 scaled by 10 is 0.99
	dropout1_rate = 0.5,              # [0, 0.5]
	dropout2_rate = None,
	weight_decay = 0,                 # doesn't work
	max_epochs = 10,                  # it terminates when overfitting or increasing, so just leave high
	output_nonlinearity = 'softmax',  # just keep softmax
	auto_stopping = True,             # stop training automatically if it seems to be failing
	outlier_frac = 0.06,              # which fraction of each class to remove as outliers
	normalize_log = True,             # use logarithm for normalization
)

#net = load_net(join(NNET_STATE_DIR, 'init_150_80.net'))

print '>> training network'
out = net.fit(train, classes - 1)

print '>> predicting test data'
prediction = net.predict_proba(test)

print '>> making submission file'
make_submission(prediction, fname = join(SUBMISSIONS_DIR, 'singlebig.csv'), digits = 8)

print '>> plotting training progress'
fig, ax = show_train_progress(net)

print '>> done!'

if __name__ == '__main__':
	show()


