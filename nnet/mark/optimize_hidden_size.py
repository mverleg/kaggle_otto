
from os.path import basename, splitext
from nnet.make_net import make_net
from nnet.prepare import normalize_data
from utils.loading import get_training_data
from utils.outliers import filter_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer


train_data, true_classes, features = get_training_data()
validator = SampleCrossValidator(train_data, true_classes, rounds = 5, test_frac = 0.2, use_data_frac = 1)
optimizer = GridOptimizer(validator = validator, use_caching = True,
	name = 'dense_sizes',
	dense1_size = [30, 25, 80, 120, 180],
	dense1_nonlinearity = 'leaky20',
	dense1_init = 'orthogonal',
	dense2_size = [30, 25, 80, 120, 180],
	dense2_nonlinearity = 'leaky20',
	dense2_init = 'orthogonal',
	learning_rate_start = 0.001,
	learning_rate_end = 0.00001,
	momentum_start = 0.9,
	momentum_end = 0.999,
	dropout1_rate = 0.5,
	dropout2_rate = None,
	weight_decay = 0,
	max_epochs = 2500,
	normalize_use_log = True,
	outlier_method = 'OCSVM'
)
for parameters, train, classes, test in optimizer.yield_batches():
	train, classes = filter_data(train, classes, cut_outlier_frac = 0.02, method = parameters.pop('outlier_method'))
	train = normalize_data(train, use_log = parameters.pop('normalize_use_log'))[0]
	net = make_net(**parameters)
	out = net.fit(train, classes - 1)
	prediction = net.predict_proba(test)
	optimizer.register_results(prediction)
optimizer.print_plot_results(save_fig_basename = splitext(basename(__file__))[0])


