
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
	name = 'singlebig',
	dense1_size = 150,
	dense1_nonlinearity = 'tanh',
	dense1_init = 'orthogonal',
	dense2_size = 80,
	dense2_nonlinearity = 'tanh',
	dense2_init = 'orthogonal',
	learning_rate_start = 0.001,
	learning_rate_end = 0.00001,
	momentum_start = 0.9,
	momentum_end = 0.999,
	dropout1_rate = 0.5,
	dropout2_rate = None,
	weight_decay = 0,
	max_epochs = 3000
)

net = load_net(join(NNET_STATE_DIR, 'init_150_80.net'))

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


