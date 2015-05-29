from scipy.stats import randint
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics.scorer import log_loss_scorer

from nnet.oldstyle.base_optimize import name_from_file
from nnet.scikit import NNet
from utils.loading import get_training_data


train, labels = get_training_data()[:2]
#test = get_testing_data()[0]
#train, labels = expand_from_test(train, labels, test, confidence = 0.9)
#gen = PositiveSparseRowFeatureGenerator()
#train = gen.fit_transform(train, labels)
#test = gen.transform(test, labels)

#random = RandomState()

opt = RandomizedSearchCV(
	estimator = NNet(
		name = name_from_file(),
		auto_stopping = True,
		max_epochs = 1500,  # binom(n = 4000, p = 0.25)
	),
	param_distributions = {
		'dense1_size': randint(low = 100, high = 1200),
		'dense2_size': randint(low = 50, high = 900),
		'dense3_size': randint(low = 25, high = 700),
	},
	fit_params = {
	},
	n_iter = 600,
	n_jobs = cpus - 1,
	scoring = log_loss_scorer,
	refit = False,
	pre_dispatch = 3,
	cv = ShuffleSplit(
		n = train.shape[0],
		n_iter = 10,
		test_size = 0.2,
		#random_state = random,
	),
	#random_state = random,
	verbose = True,
)

opt.fit(train, labels)

