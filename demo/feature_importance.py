from matplotlib.pyplot import subplots, show, setp
from numpy import ones
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from utils.features import PositiveSparseRowFeatureGenerator, PositiveSparseFeatureGenerator, DistanceFeatureGenerator
from utils.loading import get_training_data
from utils.manifold import LLEFeatures, TSNEFeatures, MDFeatures, SEFeatures
from utils.shuffling import shuffle


data, labels, features = get_training_data()
data, labels = shuffle(data, labels)[:2]
data, labels = data[:3000, :], labels[:3000]


features = sum((
	list(features),
	['avg_{0:2d}'.format(k) for k in range(20)],
	['xor_{0:2d}'.format(k) for k in range(20)],
	['dif_{0:2d}'.format(k) for k in range(20)],
	['sum', 'max', 'argmax', 'maxp_1', 'maxp_2', 'maxp_3', 'maxp_4', 'maxp_5', 'iv_0', 'iv_1', 'iv_2', 'iv_3', 'iv_4_7', 'iv_8_15', 'iv_16_30', 'iv_31_70', 'iv_70p'],
	['L1_cls{0:2d}'.format(k + 1) for k in range(9)],
	['L2_cls{0:2d}'.format(k + 1) for k in range(9)],
	['LLE{0:d}'.format(k) for k in range(10)],
	['TSNE{0:d}'.format(k) for k in range(10)],
	['MD{0:d}'.format(k) for k in range(10)],
	['SE{0:d}'.format(k) for k in range(10)],
), [])


estimator = Pipeline([
	('avg', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 20, operation_probs = (1, 0, 0))),
	('xor', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 20, operation_probs = (0, 1, 0))),
	('dif', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 20, operation_probs = (0, 0, 1))),
	('row', PositiveSparseRowFeatureGenerator()),
	('dL1', DistanceFeatureGenerator(n_neighbors = 5, distance_p = 1)),
	('dL2', DistanceFeatureGenerator(n_neighbors = 5, distance_p = 2)),
	#('LLE', LLEFeatures(extra_featurs = 10)),
	#('TSNE', TSNEFeatures(extra_featurs = 10)),
	#('MD', MDFeatures(extra_featurs = 10)),
	('SE', SEFeatures(extra_featurs = 10)),
	('csf', ExtraTreesClassifier(n_estimators = 1000)),
])


estimator.fit(data, labels)
print 'features', len(features)

importances = estimator.steps[-1][-1].feature_importances_

Q = 57
fig, axi = subplots(4, figsize = (8, 7))
fig.tight_layout()
fig.subplots_adjust(top = .95, bottom = .08, left = .02, right = .98)
for k, ax in enumerate(axi):
	x = range(len(importances))[k*Q:(k+1)*Q]
	ax.bar(x, importances[k*Q:(k+1)*Q])
	ax.set_xticks(x)
	ticks = ax.set_xticklabels(features[k*Q:(k+1)*Q])
	setp(ticks, rotation = 40, fontsize = 8)
	ax.set_xlim([k*Q, (k+1)*Q])
	ax.set_yticks([])
	ax.set_ylim([0, max(importances)])
axi[0].set_title('Feature importance (RF)')


if __name__ == '__main__':
	show()


