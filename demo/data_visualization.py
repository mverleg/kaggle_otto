
from numpy import histogram, prod
from matplotlib.pyplot import subplots, show
from utils.loading import get_training_data


"""
	Show class balance.
"""
train_data, classes, features = get_training_data()
clssizes = histogram(classes, bins = [q + .5 for q in range(10)])[0]
fig1, ax1 = subplots(figsize = (6, 4))
ax1.bar(range(1, 10), clssizes)
ax1.set_title('class frequencies')

vec_data = train_data.reshape((prod(train_data.shape),))
fig2, ax2 = subplots(figsize = (5, 4))
ax2.hist(vec_data)

if __name__ == '__main__':
	show()


