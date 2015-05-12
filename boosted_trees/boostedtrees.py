import graphlab as gl
import logging
logging.disable(logging.INFO)
import pandas as pd
import numpy as np
from utils.outliers import filter_data
from utils.postprocess import rescale_prior


def boostedTrees(train, 
                 labels, 
                 test, 
                 column_names = None, 
                 target = 'target',
                 max_iterations = 200, 
                 min_child_weight = 5, 
                 step_size = 0.2, 
                 max_depth = 10, 
                 class_weights = None, 
                 min_loss_reduction = 0.5,
                 verbose = 0,
                 outlier_frac=0.0,
                 outlier_method='EE',
                 rescale_pred=False):
    """
    train, labels, test are numpy matrices containing tha data 
    column_names is a list of column names of the test/train data
    target is the column name of the labels column
    Because it's graphlab and not sklearn, the calibration is not implemented (it's possible, but harder)
    Also, seemingly, setting sample weights is also not supported by graphlab
    """
    print outlier_frac, outlier_method, rescale_pred
    if outlier_frac > 0:
        train, labels = filter_data(train, labels, cut_outlier_frac = outlier_frac, method = outlier_method, use_caching=False)  # remove ourliers
    if column_names is None:
        column_names = range(np.shape(train)[1])
    target = 'target'
    newTrain = np.vstack((train.T, labels)).T
    pdTrain = pd.DataFrame(newTrain, columns = np.append(column_names,target))
    trainFrame = gl.SFrame(pdTrain)
    del newTrain, pdTrain
    pdTest = pd.DataFrame(test, columns = column_names)
    testFrame = gl.SFrame(pdTest)
    del pdTest
    model = gl.boosted_trees_classifier.create(trainFrame, 
                                               target=target, 
                                               max_iterations=max_iterations, 
                                               min_child_weight=min_child_weight,
                                               step_size = step_size,
                                               max_depth = max_depth,
                                               class_weights = class_weights,
                                               min_loss_reduction = min_loss_reduction,
                                               verbose = verbose)
    preds = model.predict_topk(testFrame, output_type='probability', k=9)
    preds['id'] = preds['id'].astype(int)
    #some hacky dataframe magic, creates Nx10 matrix (id in first column)
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '').sort('id')

    newPreds = preds.to_dataframe().values
    newPreds = newPreds[:,1:] #remove the id column
    del preds, model
    
    assert np.shape(newPreds)[0] == np.shape(test)[0], "conversion failed somewhere, size doesn't match"
    
    if rescale_pred:
        newPreds = rescale_prior(newPreds, np.bincount(labels))
    return newPreds  
