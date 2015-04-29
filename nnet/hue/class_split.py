



def class_split(train, train_labels):
	clsdata = []
	clslabels = []
	for cls in range(1, 10):
		index = train_labels == cls
		clsdata.append(train[index])
		clslabels.append(train_labels[index])
	return clsdata, clslabels


