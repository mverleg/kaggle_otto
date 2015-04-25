
from __future__ import division
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot, subplot2grid
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from numpy import arange
from matplotlib import gridspec


def extract_1D_data(results, labels, values):
	assert len(results.shape) == 3
	assert results.shape[-1] >= 3
	logloss_mean = results[:, :, 0].mean(1)
	logloss_std = results[:, :, 0].std(1)
	accuracy_mean = 100 * results[:, :, 1].mean(1)
	accuracy_std = 100 * results[:, :, 1].std(1)
	time_mean = results[:, :, 2].mean(1)
	time_std = results[:, :, 2].std(1)
	return logloss_mean, logloss_std, accuracy_mean, accuracy_std, time_mean, time_std


def compare_1D_3axis(plot_func, labels):

	fig = figure()
	ax1 = host_subplot(111, axes_class = AA.Axes, figsize = (8, 8))
	fig.subplots_adjust(right = 0.75)

	ax2 = ax1.twinx()
	ax3 = ax1.twinx()

	new_fixed_axis = ax3.get_grid_helper().new_fixed_axis
	ax3.axis['right'] = new_fixed_axis(loc='right', axes = ax3, offset = (60, 0))
	ax3.axis['right'].toggle(all = True)

	ax1.set_xlabel(labels[0].replace('_', ' '))
	ax1.set_ylabel('Logloss')
	ax2.set_ylabel('Accuracy (%)')
	ax3.set_ylabel('Time (s)')

	plot_func(ax1, ax2, ax3)

	ax1.legend(loc = 'lower right')

	ax1.axis['left'].label.set_color('blue')
	ax2.axis['right'].label.set_color('red')
	ax3.axis['right'].label.set_color('green')


def compare_plot(results, labels, values):
	"""
		Compare logloss, accuracy and duration for one parameter in a 3-axis logloss plot with standard deviation bars. Only for numerical parameters.

		http://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales
	"""
	def plot_handler(ax1, ax2, ax3):
		logloss_mean, logloss_std, accuracy_mean, accuracy_std, time_mean, time_std = extract_1D_data(results, labels, values)
		if min(values[0]) / max(values[0]) <= 0.01:
			ax1.set_xscale('log')
		ax1.grid()
		ax1.errorbar(x = values[0], y = list(logloss_mean), yerr = logloss_std, c = 'blue', label = 'logloss')
		ax2.errorbar(x = values[0], y = accuracy_mean, yerr = accuracy_std, c = 'green', label = 'accuracy')
		ax3.errorbar(x = values[0], y = time_mean, yerr = time_std, c = 'red', label = 'time')

	compare_1D_3axis(plot_func = plot_handler, labels = labels)


def compare_bars(results, labels, values):
	"""
		Compare logloss, accuracy and duration for one parameter in a barplot.
	"""
	def bars_handler(ax1, ax2, ax3):
		logloss_mean, logloss_std, accuracy_mean, accuracy_std, time_mean, time_std = extract_1D_data(results, labels, values)
		x = arange(len(logloss_mean))
		ax1.bar(x - 0.3, list(logloss_mean), width = 0.25, color = 'blue', label = 'logloss')
		ax2.bar(x + 0.0, accuracy_mean, width = 0.25, color = 'green', label = 'accuracy')
		ax3.bar(x + 0.3, time_mean, width = 0.25, color = 'red', label = 'time')
		ax1.set_xticks(x)
		ax1.xaxis.set_ticklabels([str(v) for v in values[0]])

	compare_1D_3axis(plot_func = bars_handler, labels = labels)


def compare_surface(results, labels, values):
	"""
		Compare logloss for two parameters in a barplot.
	"""
	assert len(results.shape) == 4
	assert results.shape[-1] >= 3
	logloss_mean = results[:, :, :, 0].mean(2)
	accuracy_mean = 100 * results[:, :, :, 1].mean(2)
	time_mean = results[:, :, :, 2].mean(2)

	fig = figure(figsize = (10, 7))
	gs = GridSpec(2, 3)
	ax1 = fig.add_subplot(gs[:2, :2])
	ax2 = fig.add_subplot(gs[0, 2])
	ax3 = fig.add_subplot(gs[1, 2])

	fig.tight_layout()

	cdict = {
		'red':   ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
		'green': ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
		'blue':  ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
	}
	redgreen = LinearSegmentedColormap('redgreen', cdict)
	invcdict = {
		'red':   ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
		'green': ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
		'blue':  ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
	}
	invredgreen = LinearSegmentedColormap('redgreen', invcdict)

	im1 = ax1.imshow(logloss_mean, cmap = redgreen, interpolation = 'none')

	ax2.imshow(accuracy_mean, cmap = invredgreen, interpolation = 'none')

	ax3.imshow(time_mean, cmap = invredgreen, interpolation = 'none')

	fig.colorbar(ax = ax1, mappable = im1, orientation = 'horizontal', label = 'values are for logloss, green is better everywhere')

	ax1.set_xlabel(labels[1].replace('_', ' '))
	ax1.set_ylabel(labels[0].replace('_', ' '))
	ax1.set_xticks(range(len(values[1])))
	ax1.set_yticks(range(len(values[0])))
	ax1.xaxis.set_ticklabels([str(v) for v in values[1]])
	ax1.yaxis.set_ticklabels([str(v) for v in values[0]])

	ax2.set_xticks([])
	ax2.set_yticks([])
	ax3.set_xticks([])
	ax3.set_yticks([])

	ax1.set_title('Logloss')
	ax2.set_title('Accuracy')
	ax3.set_title('Time')


