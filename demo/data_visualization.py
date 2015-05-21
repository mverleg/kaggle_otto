
from numpy import histogram, prod, log2, log
from matplotlib.pyplot import subplots, show
from utils.loading import get_training_data


train_data, classes, features = get_training_data()


"""
	Show class balance.
"""
clssizes = histogram(classes, bins = [q + .5 for q in range(10)])[0]
print 'class sizes:', clssizes
fig1, ax1 = subplots(figsize = (6, 4))
ax1.bar(range(1, 10), clssizes)
ax1.set_title('class frequencies')


"""
	Show value distribution / sparsity.
"""
vec_data = train_data.reshape((prod(train_data.shape),))
fig2, ax2 = subplots(figsize = (5, 4))
binned_data = histogram(vec_data, bins = range(400))[0]
ax2.bar(range(400 - 1), log2(1 + binned_data), edgecolor = 'blue')
ax2.set_ylabel('$log_2(1 + \#)$')
ax3 = fig2.add_subplot(2, 2, 2)
ax3.pie(x = [(vec_data == 0).sum(), (vec_data != 0).sum()], explode = [0.15, 0], labels = ['0', '>0'], colors = ['red', 'blue'])

fig3, ax3 = subplots(figsize = (5, 4))
nonzero_per_sample = (train_data != 0).sum(1)
ax3.hist(nonzero_per_sample, bins = range(70))
ax3.set_xlabel('Nonzero elements per sample')
ax3.set_ylabel('#occurences')


if __name__ == '__main__':
	show()


