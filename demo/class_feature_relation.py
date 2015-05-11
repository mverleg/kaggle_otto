
from numpy import zeros, abs, logical_xor
from settings import NCLASSES, NFEATS
from utils.loading import get_training_data
from matplotlib.pyplot import subplots, show


def class_feature_sum_count(train, labels):
	sm, cnt = zeros((NFEATS, NCLASSES)), zeros((NFEATS, NCLASSES))
	for cls in range(0, NCLASSES):
		sm[:, cls] = train[cls + 1 == labels].sum(0)
		cnt[:, cls] = (train[cls + 1 == labels] != 0).sum(0)

	fig, (ax_sm, ax_cnt) = subplots(2, figsize = (8, 3))
	ax_sm.imshow(sm.T, interpolation = 'none')
	ax_sm.set_title('Class features sum')
	ax_cnt.imshow(cnt.T, interpolation = 'none')
	ax_cnt.set_title('Class features count')

	return sm,cnt


# def biclass_distance(train, labels, cls1, cls2):
# 	l1, xr = zeros((NFEATS,)), zeros((NFEATS,))
# 	for cls in range(0, NCLASSES):
# 		l1[cls] = abs(train[labels == cls1] - train[labels == cls2]).sum(0)
# 		#xr[cls] = logical_xor(abs(train[labels == cls1], train[labels == cls2]))
#
# 	fig, (ax_l1, ax_xr) = subplots(2, figsize = (8, 3))
# 	ax_l1.bar(range(1, NFEATS + 1), l1)
# 	ax_l1.set_title('L1 distance between {0:d} and {1:d}'.format(cls1, cls2))
# 	#ax_cnt.imshow(cnt.T, interpolation = 'none')
# 	#ax_cnt.set_title('Class features count')


if __name__ == '__main__':
	train, labels = get_training_data()[:2]
	class_feature_sum_count(train, labels)
	show()


