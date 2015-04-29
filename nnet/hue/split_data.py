

def split_data(data, labels, test_frac = 0.1):
	N = int(len(labels) * test_frac)
	train = data[N:, :]
	test = data[:N, :]
	train_labels = labels[N:]
	test_labels = labels[:N]
	return train, train_labels, test, test_labels


