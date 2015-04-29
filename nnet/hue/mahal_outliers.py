import numpy as np
from numpy import cov, array, copy, logical_not
from scipy.spatial.distance import mahalanobis


def remove_outliers_mahalanobis(class_set, class_set_labels, offset = 0.95):
	class_set, class_set_labels = copy(class_set), copy(class_set_labels)
	class_sizes = []
	# outlier_data = np.array([]).reshape(0,93)
	outlier_data = np.empty((0,93))
	outlier_labels = np.empty((0))
	cls = 0
	for class_data, class_labels in zip(class_set, class_set_labels):
		cls += 1
		mean = class_data.mean(0)
		not_inverse_covariance = cov(class_data.T)
		distances = array([mahalanobis(sample, mean, not_inverse_covariance) for sample in class_data])
		if offset < 1:
			#print 'using fixed percentage cutoff', offset
			cutoff_value = sorted(distances)[int(offset * len(distances))]
		else:
			#print 'using threshold', offset
			cutoff_value = offset
		#print 'curring', cls, 'at', cutoff_value
		index = distances < cutoff_value
		class_sizes.append(index.sum())
		#print ' before size', class_data.shape[0]
		class_set[cls - 1] = class_data[index, :]
		class_set_labels[cls - 1] = class_labels[index]
		# outlier_data.append(class_data[logical_not(index), :])
		# outlier_labels.append(class_labels[logical_not(index)])
		outlier_data = np.vstack((outlier_data, class_data[logical_not(index), :]))
		outlier_labels = np.append(outlier_labels, class_labels[logical_not(index)])
		# outlier_labels = np.vstack((outlier_labels, np.transpose(class_labels[logical_not(index)])))
		#print ' after size', class_data[index, :].shape[0]
	return class_set, class_set_labels, outlier_data, outlier_labels, min(class_sizes)


