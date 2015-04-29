
# you should run the code from the main directory (F7 in pycharm)

from numpy import savetxt, isnan
from utils.loading import get_training_data, get_testing_data


# load the trian and test data without any shuffling / normalziing / filtering / ...
train_data, true_labels = get_training_data()[:2]
test_data = get_testing_data()[0]

print 'number of NaN values in training:', isnan(train_data).sum()
print 'number of NaN values in testing:', isnan(test_data).sum()

# save it to data directory
print 'saving training data as csv'
savetxt('nnet/hue/data/train_noheader.csv', train_data, delimiter = ',', fmt = '%d')
savetxt('nnet/hue/data/labels.csv', true_labels, delimiter = ',', fmt = '%d')
print 'saving testing data as csv'
savetxt('nnet/hue/data/test_noheader.csv', test_data, delimiter = ',', fmt = '%d')


