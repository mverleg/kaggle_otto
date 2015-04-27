
from os.path import join
from nnet.make_net import make_net
from nnet.visualize import show_train_progress
from settings import AUTO_IMAGES_DIR
from utils.outliers import get_filtered_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer


# try with and without logscale
# try with EE and OCSVM
train_data, true_classes, features = get_filtered_data(cut_outlier_frac = 0.05, filepath = TRAIN_DATA_PATH, method = 'EE', logscale = False)
validator = SampleCrossValidator(train_data, true_classes, rounds = 1, test_frac = 0.2, use_data_frac = 1)
optimizer = GridOptimizer(validator = validator, use_caching = True,
	name = 'hidden1_size',
	hidden1_size = [15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 180],
	hidden1_nonlinearity = 'tanh',
	hidden1_init = 'orthogonal',
	learning_rate_start = 0.001,
	learning_rate_end = 0.00001,
	momentum_start = 0.9,
	momentum_end = 0.999,
	max_epochs = 3000,
	dropout_rate = None,
)
for parameters, train, classes, test in optimizer.yield_batches():
	net = make_net(**parameters)
	out = net.fit(train, classes - 1)
	prediction = net.predict_proba(test)
	optimizer.register_results(prediction)
	fig, ax = show_train_progress(net)
	fig.savefig(join(AUTO_IMAGES_DIR, '{0:s}_{1:d}.png'.format(parameters['name'], parameters['hidden1_size'])))
optimizer.print_plot_results()


