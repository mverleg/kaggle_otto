
from multiprocessing import cpu_count
from scipy.stats import binom, norm, triang, randint
from numpy.random import RandomState
from sklearn.cross_validation import KFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from nnet.base_optimize import name_from_file
from nnet.prepare import LogTransform
from nnet.scikit import NNet
from utils.loading import get_training_data, get_testing_data
from settings import SEED
from utils.expand_train import expand_from_test
from utils.features import PositiveSparseFeatureGenerator, PositiveSparseRowFeatureGenerator


# from matplotlib.pyplot import subplots, show
# fig, (ax1, ax2, ax3) = subplots(3)
# ax1.hist(triang(loc = 0, c = 0, scale = 1).rvs(size = 5000))
# ax2.hist(binom(n = 256, p = 0.5).rvs(size = 5000))
# ax3.hist(norm(0.001, 0.001).rvs(size = 5000))
# show()

train, labels = get_training_data()[:2]
test = get_testing_data()[0]
train, labels = expand_from_test(train, labels, test, confidence = 0.9)
gen = PositiveSparseRowFeatureGenerator()
train = gen.fit_transform(train, labels)
test = gen.transform(test, labels)

cpus = max(cpu_count() - 1, 1)
random = RandomState()

opt = RandomizedSearchCV(
	estimator = Pipeline([
		('gen1', PositiveSparseFeatureGenerator(difficult_classes = (2, 3), extra_features = 50)),
		('gen2', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 50)),
		('gen3', PositiveSparseFeatureGenerator(difficult_classes = (1, 9), extra_features = 63)),
		('log', LogTransform()),
		('scale', MinMaxScaler(feature_range = (0, 3))),
		('nn', NNet(
			name = name_from_file(),
			dense1_nonlinearity = 'rectify',
			dense1_init = 'glorot_normal',
			#'nn__save_snapshots_stepsize': None,
			auto_stopping = True,
			dropout2_rate = None,
			dropout3_rate = None,
			max_epochs = 1500,  # binom(n = 4000, p = 0.25)
			weight_decay = 0,
		)),
	]),
	param_distributions = {
		'nn__batch_size': binom(n = 256, p = 0.5),
		'nn__learning_rate': norm(0.0005, 0.0005),
		'nn__learning_rate_scaling': [1, 10, 100, 1000],
		'nn__momentum': [0, 0.9, 0.99, 0.999],
		'nn__momentum_scaling': [1, 10, 100],
		'nn__dense1_size': randint(low = 100, high = 1200),
		'nn__dense2_size': randint(low = 50, high = 900),
		'nn__dense3_size': randint(low = 25, high = 700),
		'nn__dropout0_rate': triang(loc = 0, c = 0, scale = 1),  # beta(a = 0.5, b = 0.5),
		'nn__dropout1_rate': triang(loc = 0, c = 0, scale = 1),
	},
	fit_params = {
	},
	n_iter = 10,
	n_jobs = cpus,
	scoring = None,
	iid = False,
	refit = False,
	pre_dispatch = cpus + 2,
	cv = KFold( #todo: use less validation data
		n = train.shape[0],
		n_folds = 3, shuffle = True,
		random_state = random
	),
	random_state = random,
)
print opt
print opt.fit(train, labels)

#prediction = scale_to_priors(prediction, priors = bincount(labels)[1:] / float64(len(labels)))


