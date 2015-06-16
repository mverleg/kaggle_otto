
from sklearn.pipeline import Pipeline
from utils.features import PositiveSparseRowFeatureGenerator, PositiveSparseFeatureGenerator, DistanceFeatureGenerator
from utils.loading import get_training_data
from utils.shuffling import shuffle


data, labels = get_training_data()[:2]
data, labels = shuffle(data, labels)[:2]
data, labels = data[0, :1000], labels[:1000]


estimator = Pipeline([
	('avg', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 20, operation_probs = (1, 0, 0)),
	('xor', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 20, operation_probs = (0, 1, 0)),
	('dif', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 20, operation_probs = (0, 0, 1)),
	('row', PositiveSparseRowFeatureGenerator()),
	('dL1', DistanceFeatureGenerator(n_neighbors = 5, distance_p = 1)),
	('dL2', DistanceFeatureGenerator(n_neighbors = 5, distance_p = 2)),
	('csf', ),
])


