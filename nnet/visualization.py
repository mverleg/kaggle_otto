
from matplotlib.pyplot import subplots
from numpy import array


def get_train_progress(net):
	return [
		array([res['epoch'] for res in net.train_history_]),
		array([res['train_loss'] for res in net.train_history_]),
		array([res['valid_loss'] for res in net.train_history_]),
	]


def show_train_progress(net):

	epochs, train_loss, valid_loss = get_train_progress(net)

	fig, ax = subplots()
	ax.plot(epochs, train_loss, color = 'blue', label = 'train')
	ax.plot(epochs, valid_loss, color = 'green', label = 'test')
	ax.set_xlabel('Log epoch')
	ax.set_ylabel('Loss')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.legend()

	return fig, ax


