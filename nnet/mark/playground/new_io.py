
# Due to this error:
#  ValueError: Loading weights from a list of parameter values is no longer supported.
#  Please send me something like the return value of 'net.get_all_param_values()' instead.
# testing new method

import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from nnet.prepare import LogTransform
from nnet.scikit import NNet
from utils.loading import get_preproc_data

warnings.filterwarnings("ignore")

train, labels, test = get_preproc_data(Pipeline([
	('log', LogTransform()),
	('scale03', MinMaxScaler(feature_range = (0, 3))),
]), expand_confidence = 0.9)

nn = NNet(max_epochs = 3)
nn.fit(train, labels)
nn.save(filepath = '/tmp/test')
nn = NNet.load(filepath = '/tmp/test')


