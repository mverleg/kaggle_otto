
from numpy import zeros
from sklearn.neighbors import KNeighborsClassifier
from settings import NCLASSES
from utils.features import DistanceFeatureGenerator
from utils.loading import get_training_data

# recommended to run with -vv

train_data, true_labels = get_training_data()[:2]

knn = [None] * NCLASSES
for cls in range(1, NCLASSES + 1):
	knn[cls - 1] = KNeighborsClassifier(
		n_neighbors = 10,
		p = 1,
	)
	f = true_labels == cls
	knn[cls - 1].fit(train_data[f], zeros((f.sum(),)))

try_data = train_data[::300, :]
try_labels = true_labels[::300]

gen = DistanceFeatureGenerator()
gen.fit(train_data, true_labels)
new_data = gen.transform(try_data)

print new_data.shape


