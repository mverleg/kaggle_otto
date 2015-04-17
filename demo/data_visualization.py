
from numpy import histogram
from matplotlib.pyplot import subplots, show
from utils.loading import get_training_data


"""
	Show class balance.
"""
train_data, classes, features = get_training_data()
clssizes = histogram(classes, bins = [q + .5 for q in range(10)])[0]
fig, ax = subplots(figsize = (6, 4))
ax.bar(range(1, 10), clssizes)
ax.set_title('class frequencies')


if __name__ == '__main__':
	show()


