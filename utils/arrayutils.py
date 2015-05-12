import numpy as np

def stack_predictions(predictionmatrix2d, Q):
	"""
	Transforms a NxQC matrix into a QxNxC matrix, by stacking the slices
		across axis 0. The inverse of unstack_predictions
	:param predictionmatrix2d: The NxQC matrix
	:param Q: Q, needed to split the matrix correctly.
	"""
	return np.array(np.hsplit(predictionmatrix2d, Q))

def unstack_predictions(predictionmatrix3d, retQ = False):
	"""
	Transforms a QxNxC matrix into a NxQC matrix, by horizontally stacking the matrices
		that were originally stacked across axis 0. The inverse of stack_predictions
	:param predictionmatrix3d: The QxNxC matrix
	:param retQ: Whether or not to return Q as well, False by default
	:return: NxQC matrix obtained by unstacking the input matrix
	"""
	Q, _, _ = np.shape(predictionmatrix3d)
	unstacked = np.hstack(np.squeeze(np.vsplit(predictionmatrix3d,Q)))
	if retQ:
		return unstacked, Q
	return unstacked
