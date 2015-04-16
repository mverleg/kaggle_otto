
from utils.loading import get_training_data
from utils.shuffling import shuffle
from time import time


"""
	Time the shuffle function for training data.
"""
train_data, classes, features = get_training_data()
t = time()
sdata, sclasses, skey = shuffle(train_data, classes = classes)
print 'Shuffling training data takes {0:.4f}s'.format(time() - t)


