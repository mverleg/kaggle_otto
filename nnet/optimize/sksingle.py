from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.metrics.scorer import log_loss_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from nnet.oldstyle.base_optimize import name_from_file
from nnet.prepare import LogTransform
from nnet.scikit import NNet
from utils.features import PositiveSparseRowFeatureGenerator, DistanceFeatureGenerator, PositiveSparseFeatureGenerator
from utils.loading import get_preproc_data


train, labels, test = get_preproc_data(Pipeline([
	('row', PositiveSparseRowFeatureGenerator()),
	('distp31', DistanceFeatureGenerator(n_neighbors = 3, distance_p = 1)),
	('distp52', DistanceFeatureGenerator(n_neighbors = 5, distance_p = 2)),
	('gen23', PositiveSparseFeatureGenerator(difficult_classes = (2, 3), extra_features = 40)),
	('gen234', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 40)),
	('gen19', PositiveSparseFeatureGenerator(difficult_classes = (1, 9), extra_features = 40)),
	('log', LogTransform()),
	('scale03', MinMaxScaler(feature_range = (0, 3))),
]), expand_confidence = 0.9)

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

cv = ShuffleSplit(
	n = train.shape[0],
	n_iter = 5,
	test_size = 0.2,
)

print cross_val_score(net, train, labels, scoring = log_loss_scorer, cv = cv, n_jobs = 1, verbose = True)


