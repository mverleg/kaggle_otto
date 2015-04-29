
from nnet.hue.preprocessing2 import preprocess
from nnet.hue.train import do_train


class_set_1, class_set_labels_1, outlier_data_1, outlier_labels_1, smallest_size_1, class_set_2, \
	class_set_labels_2, outlier_data_2, outlier_labels_2, smallest_size_2, test, test_labels = preprocess(test_frac = 0.1)

print 'first set'
do_train(smallest_size_1, class_set_1, class_set_labels_1, outlier_data_1, outlier_labels_1, test, test_labels, round_nr = 5)

print 'second set'
do_train(smallest_size_2, class_set_2, class_set_labels_2, outlier_data_2, outlier_labels_2, test, test_labels, round_nr = 5)


