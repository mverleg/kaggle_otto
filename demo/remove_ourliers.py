
from random import sample
from matplotlib.cm import get_cmap
from matplotlib.pyplot import subplots, show
from numpy import array, log2
from sklearn.decomposition import PCA
from settings import NCLASSES
from utils.outliers import get_filtered_data


train_data, true_classes, features = get_filtered_data(cut_outlier_frac = 0.08)
train_data = log2(1 + train_data)

""" Data subset """
filter = array(sample(range(train_data.shape[0]), 500))
train_data = train_data[filter, :]
true_classes = true_classes[filter]

""" Get PCA """
transformed = PCA(n_components = 2, copy = True, whiten = False).fit(train_data).transform(train_data)

""" Visualize """
cmap = get_cmap('gist_rainbow')
fig, ax = subplots(figsize = (12, 10))
fig.tight_layout()
for cls in range(1, NCLASSES + 1):
	ax.scatter(transformed[true_classes == cls, 0], transformed[true_classes == cls, 1], c = cmap((cls - 1.) / 9), label = str(cls))
ax.legend()


if __name__ == '__main__':
	show()


