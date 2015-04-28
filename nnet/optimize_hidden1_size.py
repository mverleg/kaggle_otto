
from os.path import join
from nnet.make_net import make_net
from nnet.prepare import prepare_data
from nnet.visualization import show_train_progress
from settings import AUTO_IMAGES_DIR
from utils.loading import get_training_data
from utils.outliers import filter_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer


# try with and without logscale
# try with EE and OCSVM
train_data, true_classes, features = get_training_data()
train_data = prepare_data(train_data)
validator = SampleCrossValidator(train_data, true_classes, rounds = 1, test_frac = 0.2, use_data_frac = 1)
optimizer = GridOptimizer(validator = validator, use_caching = True,
	name = 'hidden1_size',
	dense1_size = [15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 180],
	dense1_nonlinearity = 'tanh',
	dense1_init = 'orthogonal',
	dense2_size = None,
	dense2_nonlinearity = 'tanh',
	dense2_init = 'orthogonal',
	learning_rate_start = 0.001,
	learning_rate_end = 0.00001,
	momentum_start = 0.9,
	momentum_end = 0.999,
	dropout1_rate = None,
	dropout2_rate = None,
	weight_decay = 0,
)
for parameters, train, classes, test in optimizer.yield_batches():
	train, classes = filter_data(train, classes, cut_outlier_frac = 0.06, method = 'OCSVM')
	net = make_net(max_epochs = 200 + 15 * parameters['dense1_size'], **parameters)  # dynamic epoch count, only for 1 layer network
	out = net.fit(train, classes - 1)
	prediction = net.predict_proba(test)
	optimizer.register_results(prediction)
	fig, ax = show_train_progress(net)
	fig.savefig(join(AUTO_IMAGES_DIR, '{0:s}_{1:d}.png'.format(parameters['name'], parameters['hidden1_size'])))
optimizer.print_plot_results()


