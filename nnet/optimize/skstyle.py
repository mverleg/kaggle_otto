
from multiprocessing import cpu_count
from scipy.stats import binom, norm
from numpy.random import RandomState
from sklearn.cross_validation import KFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from nnet.prepare import LogTransform
from nnet.scikit import NNet
from utils.loading import get_training_data, get_testing_data
from settings import SEED
from utils.expand_train import expand_from_test
from utils.features import PositiveSparseFeatureGenerator, PositiveSparseRowFeatureGenerator


train, labels = get_training_data()[:2]
test = get_testing_data()[0]
train, labels = expand_from_test(train, labels, test, confidence = 0.9)
gen = PositiveSparseRowFeatureGenerator()
train = gen.fit_transform(train, labels)
test = gen.transform(test, labels)

cpus = max(cpu_count() - 1, 1)
random = RandomState(SEED)

opt = RandomizedSearchCV(
	estimator = Pipeline([
		('gen1', PositiveSparseFeatureGenerator(difficult_classes = (2, 3), extra_features = 50)),
		('gen2', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 50)),
		('gen3', PositiveSparseFeatureGenerator(difficult_classes = (1, 9), extra_features = 63)),
		('log', LogTransform()),
		('scale', MinMaxScaler(feature_range = (0, 3))),
		('nn', NNet()),
	]),
	param_distributions = {
		'nn__batch_size': binom(n = 256, p = 0.5),
		'nn__learning_rate': norm(0.001, 0.002),
	},
	fit_params = {
	},
	n_iter = 10,
	n_jobs = cpus,
	scoring = None,
	iid = False,
	refit = False,
	pre_dispatch = cpus + 2,
	cv = KFold(
		n = train.shape[0],
		n_folds = 3, shuffle = True,
		random_state = random
	),
	random_state = random,
)
print opt
print opt.fit(train, labels)

#prediction = scale_to_priors(prediction, priors = bincount(labels)[1:] / float64(len(labels)))


