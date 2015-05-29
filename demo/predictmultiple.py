from os.path import basename, splitext
from sys import modules

from numpy import load, save

from gradient_boosting.gradientboosting import gradientBoosting
from random_forest.newRandomForest import randomForest
from nnet.oldstyle import make_net
from nnet.prepare import normalize_data, equalize_class_sizes
from utils.outliers import filter_data


def train_test_NN(train, classes, test, **parameters):

    train, classes = equalize_class_sizes(train, classes)
    train, classes = filter_data(train, classes, cut_outlier_frac = 0.06, method = 'OCSVM')  # remove ourliers

    train = normalize_data(train, use_log = True)[0]  # also converts to floats
    test = normalize_data(test, use_log = True)[0]

    parameters['dense2_nonlinearity'] = parameters['dense1_nonlinearity']  # hack1
    parameters['dense2_init'] = parameters['dense1_init']  # hack2
    net = make_net(**parameters)
    net.fit(train, classes - 1)
    return net.predict_proba(test)

testmat = load('data/testmat.npy').astype('uint16')
testmat = testmat[:,1:]
trainmat = load('data/trainmat.npy').astype('uint16')
trainmat = trainmat[:,1:]
trainclas = load('data/trainclas.npy')

doForest = False
doGradient = False
doNeural = True

forestparams = {
"n_estimators" : 200,
"criterion" : 'gini', #['gini', 'entropy'], #gini seems better (default)
"max_features" : 'sqrt', #['sqrt', 'log2', None], #sqrt seems good (default), None is about as good but much slower
"max_depth" : None, #[None, 75, 100, 200, 300], #at least higher than 20
"min_samples_split" : 2, #[2,4,6] #not much difference, 2 (default) seems slightly the best
"min_samples_leaf" : 1, #[1,2,3] #1 seems clearly the best (default)
"min_weight_fraction_leaf" : 0, #[0.,0.1,0.2] #0 seems clearly the best (default)
"max_leaf_nodes" : None, #[None, 5000,10000,50000] #can vary, don't do simultaneously with max_depth
"n_jobs" : 1 ,
"verbose" : 1,
"class_weight" : None, #[None, "auto"] #None seems slightly better
"calibration" : 5,#[0.05,0.1,0.15,0.2,0.25,2,3,4,5]
"calibrationmethod" : 'isotonic' ,#['sigmoid', 'isotonic']
"outlier_frac" : False, #[False, 0.03, 0.06]
"outlier_method" : 'EE', #['EE', 'OCSVM']
"undersample" : False, #[False,True]
"rescale_pred" : True,#[False,True]
"sample_weight" : None #[None, "inverted"]
}
if doForest:
    forestprediction = randomForest(trainmat, trainclas, testmat, **forestparams)
    save('results/forestprediction', forestprediction)

gradientparams = {
"learning_rate" : 0.05, #[0.05,0.075,0.1] #default 0.1
"n_estimators" : 300,
"max_depth" : 5, #5 works okay, 7 as well, default 5
"min_samples_split" : 2, #really does not matter for [1,2,3]
"min_samples_leaf" : 5, #[3,5,7]#small difference, but higher seems a little better
"min_weight_fraction_leaf" : 0, #[0,0.1,0.2] #default 0
"max_features" : 'sqrt', #['sqrt','log2',None]
"verbose" : 1,
"calibration" : False,#[0,0.1]
"calibrationmethod" : 'sigmoid', #['sigmoid', 'isotonic']
"outlier_frac" : False, #[False, 0.03, 0.06]
"outlier_method" : 'EE', #['EE', 'OCSVM']
"undersample" : False, #[False,True] #didn't try, because probably won't work anyway
"rescale_pred" : False, # [False,True]
"sample_weight" : None #[None, "inverted"]
}
if doGradient:
    gradientprediction = gradientBoosting(trainmat, trainclas, testmat, **gradientparams)
    save('results/gradientprediction', gradientprediction)


name = '{0:s}.log'.format(splitext(basename(getattr(modules['__main__'], '__file__', 'optimize.default')))[0])  # automatic based on filename
neuralparams = {
"name" : name,                      # just choose something sensible
"dense1_size" : 180 ,               # [30, 25, 80, 120, 180]  - default 60
"dense1_nonlinearity" : 'tanh', #['tanh', 'sigmoid', 'rectify', 'leaky2', 'leaky20', 'softmax'], #'leaky20'  # ['tanh', 'sigmoid', 'rectify', 'leaky2', 'leaky20', 'softmax']
"dense1_init" :  'glorot_uniform', #['orthogonal', 'sparse', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],  #'orthogonal'       # ['orthogonal', 'sparse', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
"dense2_size" : 180,                # [30, 25, 80, 120, 180]
"dense2_nonlinearity" : 'leaky20',  # this is coupled to dense1_nonlinearity through hack#1
"dense2_init" : 'orthogonal',       # idem hack2
"learning_rate" : 0.001,            # [0.1, 0.01, 0.001, 0.0001]
"learning_rate_scaling" : 10,  # 10 (used to be 100)     # [1, 10, 100]
"momentum" : 0.99 ,                  # [0, 0.9, 0.99]
"momentum_scaling" : 10 ,          # [1, 10, 100]
"dropout1_rate" : 0,                # [0, 0.5]
"dropout2_rate" : None,
"weight_decay" : 0 ,                # doesn't work
"max_epochs" : 5000,                # it terminates when overfitting or increasing, so just leave high
"output_nonlinearity" : 'softmax',  # just keep softmax
"verbosity" : 1,                    # farts output hard, but at least you get some idea of progress
"auto_stopping" : True }

if doNeural:
    neuralprediction = train_test_NN(trainmat, trainclas, testmat, **neuralparams)
    save('results/neuralprediction', neuralprediction)