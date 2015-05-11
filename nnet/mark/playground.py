
from nnet.nnio import save_knowledge, load_knowledge
from nnet.make_net import make_net
from nnet.prepare import normalize_data
from nnet.visualization import show_train_progress
from utils.loading import get_training_data
from matplotlib.pyplot import show
from utils.shuffling import shuffle


print '>> loading train data'
train, classes, features = get_training_data()

print '>> normalizing training data'
train, norm = normalize_data(train, use_log = True)

print '>> shuffling data'
train, classes, key = shuffle(train, classes)

print '>> removing outliers'
print '... skipped'
#train, classes = filter_data(train, classes, cut_outlier_frac = 0.06, method = 'OCSVM')

print '>> making network'
net = make_net(
	name = 'single',                  # just choose something sensible
	dense1_nonlinearity = 'tanh',     # ['tanh', 'sigmoid', 'rectify', 'leaky2', 'leaky20' 'softmax']
	dense1_init = 'glorot_uniform',   # ['orthogonal', 'sparse', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
	dense1_size = 180,                # [30, 25, 80, 120, 180]
	dense2_size = 180,
	dense3_size = None,
	learning_rate = 0.001,            # initial learning reate
	learning_rate_scaling = 10,       # pogression over time; 0.1 scaled by 10 is 0.01
	momentum = 0.99,                  # initial momentum
	momentum_scaling = 10,            # 0.9 scaled by 10 is 0.99
	dropout1_rate = 0.5,              # [0, 0.5]
	dropout2_rate = None,
	weight_decay = 0,                 # constrain the weights to avoid overfitting
	max_epochs = 2000,                # it terminates when overfitting or increasing, so just leave high
	output_nonlinearity = 'softmax',  # just keep softmax
	auto_stopping = True,             # stop training automatically if it seems to be failing
)

print '>> loading pretrained network'
load_knowledge(net, 'results/nnets/single_pretrain.net.npz')

print '>> training network'
out = net.fit(train, classes - 1)

print '>> saving network'
save_knowledge(net, '/home/mark/test.npz')

print '>> plotting training progress'
fig, ax = show_train_progress(net)

print '>> done!'

if __name__ == '__main__':
	show()


