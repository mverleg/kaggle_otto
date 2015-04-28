
from os.path import join, basename, splitext
from nnet.make_net import make_net
from nnet.prepare import normalize_data, equalize_class_sizes
from nnet.visualization import show_train_progress
from settings import AUTO_IMAGES_DIR
from utils.loading import get_training_data
from utils.outliers import filter_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer


# try with and without logscale
# try with EE and OCSVM
train_data, true_classes, features = get_training_data()  # load the train data
train_data, true_classes = equalize_class_sizes(train_data, true_classes)
train_data, true_classes = filter_data(train_data, true_classes, cut_outlier_frac = 0.06, method = 'OCSVM')  # remove ourliers
train_data = normalize_data(train_data, use_log = True)  # also converts to floats
validator = SampleCrossValidator(train_data, true_classes, rounds = 3, test_frac = 0.2, use_data_frac = 1)
optimizer = GridOptimizer(validator = validator, use_caching = True,
	name = 'hidden1_size',
	dense1_size = [15, 20, 25, 30],
	dense1_nonlinearity = 'tanh',
	dense1_init = 'he_normal',
	dense2_size = None,
	dense2_nonlinearity = 'tanh',
	dense2_init = 'he_normal',
	learning_rate_start = 0.001,
	learning_rate_end = 0.00001,
	momentum_start = 0.9,
	momentum_end = 0.999,
	dropout1_rate = None,
	dropout2_rate = None,
	weight_decay = 0,
	max_epochs = 1,
	output_nonlinearity = 'softmax',
)
for parameters, train, classes, test in optimizer.yield_batches():
	net = make_net(**parameters)  # dynamic epoch count, only for 1 layer network
	out = net.fit(train, classes - 1)
	prediction = net.predict_proba(test)
	optimizer.register_results(prediction)
	fig, ax = show_train_progress(net)
	fig.savefig(join(AUTO_IMAGES_DIR, '{0:s}_{1:d}_{2:d}.png'.format(parameters['name'], parameters['dense1_size'], parameters['dense2_size'] or 0)))
	fig.close()
optimizer.print_plot_results(save_fig_basename = splitext(basename(__file__))[0])


