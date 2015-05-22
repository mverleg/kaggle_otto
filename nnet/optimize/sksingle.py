
from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.metrics.scorer import log_loss_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from nnet.base_optimize import name_from_file
from nnet.prepare import LogTransform
from nnet.scikit import NNet
from utils.loading import get_training_data, get_testing_data
from utils.expand_train import expand_from_test
from utils.features import PositiveSparseFeatureGenerator, PositiveSparseRowFeatureGenerator


train, labels = get_training_data()[:2]
test = get_testing_data()[0]
train, labels = expand_from_test(train, labels, test, confidence = 0.9)
gen = PositiveSparseRowFeatureGenerator()
train = gen.fit_transform(train, labels)
test = gen.transform(test, labels)

net = NNet(
	name = name_from_file(),
	dense1_nonlinearity = 'rectify',
	dense1_init = 'glorot_normal',
	auto_stopping = True,
	max_epochs = 1000,
	batch_size = 256,
	learning_rate = 0.0005,
	learning_rate_scaling = 100,
	momentum = 0.9,
	momentum_scaling = 100,
	dense1_size = 100,
	dense2_size = 50,
	dense3_size = None,
	dropout0_rate = 0,
	dropout1_rate = 0,
	dropout2_rate = 0,
	dropout3_rate = 0,
	weight_decay = 0.001,
)

pnet = Pipeline([
	('gen1', PositiveSparseFeatureGenerator(difficult_classes = (2, 3), extra_features = 50)),
	('gen2', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 50)),
	('gen3', PositiveSparseFeatureGenerator(difficult_classes = (1, 9), extra_features = 63)),
	('log', LogTransform()),
	('scale', MinMaxScaler(feature_range = (0, 3))),
	('nn', net),
])

cv = ShuffleSplit(
	n = train.shape[0],
	n_iter = 5,
	test_size = 0.2,
)

print cross_val_score(pnet, train, labels, scoring = log_loss_scorer, cv = cv, n_jobs = 1, verbose = True)


