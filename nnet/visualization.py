
from matplotlib.pyplot import subplots
from numpy import array, savetxt
from os.path import join
from settings import AUTO_IMAGES_DIR


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


class TrainProgressPlotter(object):

	def __init__(self, base_name = 'net_hist'):
		self.base_path = join(AUTO_IMAGES_DIR, base_name)

	def __call__(self, nn, train_history):
		train = [d['train_loss'] for d in train_history]
		valid = [d['valid_loss'] for d in train_history]
		savetxt('{0:s}_{1:d}.csv'.format(self.base_path, train_history[-1]['epoch']), array([train, valid]), header = 'train, validation')
		if len(train) >= 3:
			from matplotlib.pyplot import subplots
			fig, ax = subplots(figsize = (6, 4))
			ax.plot(train, color = 'blue', label = 'train')
			ax.plot(valid, color = 'red', label = 'test')
			ax.legend()
			ax.set_xlim([0, max(10, len(train))])
			ax.set_ylim([0, 1.05 * max(max(train), max(valid))])
			fig.savefig(self.base_path + '.png')


