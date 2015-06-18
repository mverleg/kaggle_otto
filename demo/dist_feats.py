from numpy.ma import logical_and
from matplotlib.pyplot import subplots, show
from numpy import array

from nnet.prepare import LogTransform
from utils.loading import get_training_data
from utils.shuffling import shuffle
from nndist.distance import DistanceFeatureGenerator

N = 500

train, labels = get_training_data()[:2]
train = train[logical_and(5 <= labels, labels <= 7), :]
labels = labels[logical_and(5 <= labels, labels <= 7)] - 4
train, labels = shuffle(train, labels)[:2]
train = train[:N, :]
labels = labels[:N]
print train.shape, train.dtype

train = LogTransform().fit_transform(train)
print train.shape, train.dtype

gen = DistanceFeatureGenerator(n_neighbors = 3, distance_p = 2, nr_classes = 3)
train = gen.fit_transform(train, labels)
print train.shape, train.dtype

#train = MinMaxScaler(feature_range = (0, 3)).fit_transform(train)
#print train.shape, train.dtype


fig, ax = subplots(figsize = (7, 6))
colors = ['r', 'g', 'b']
for cls in range(1, 4):
	ax.scatter(train[labels == cls, -2], train[labels == cls, -1], c = colors[cls - 1], label = 'cls {0:d}'.format(cls + 4))
ax.set_xlim([0, 12])
ax.set_ylim([0, 12])
ax.set_xlabel('dist to cls 6')
ax.set_ylabel('dist to cls 7')
ax.set_title('dist features (L2) for subset of cls 5-7')
ax.legend()

fig, ax = subplots(figsize = (7, 6))
for cls in range(1, 4):
	ax.scatter(array(range(N))[labels == cls], train[labels == cls, -3], c = colors[cls - 1], label = 'cls {0:d}'.format(cls + 4))
ax.set_xlim([0, 500])
ax.set_ylabel('dist to cls 5')
ax.set_title('dist features (L2) for subset of cls 5-7')
ax.legend()

show()


