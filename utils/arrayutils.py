import numpy as np

def stack_predictions(predictionmatrix2d, Q):
   return np.array(np.hsplit(predictionmatrix2d, Q))
   
def unstack_predictions(predictionmatrix3d):
    Q, _, _ = np.shape(predictionmatrix3d)
    return np.hstack(np.squeeze(np.vsplit(predictionmatrix3d,Q)))
