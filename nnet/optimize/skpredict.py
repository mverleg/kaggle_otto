
from os.path import join
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from nnet.prepare import LogTransform
from nnet.scikit import NNet
from settings import SUBMISSIONS_DIR, PRIORS
from utils.features import PositiveSparseFeatureGenerator, PositiveSparseRowFeatureGenerator
from utils.ioutil import makeSubmission
from utils.loading import get_preproc_data
from utils.postprocess import scale_to_priors


train, labels, test = get_preproc_data(Pipeline([
	('row', PositiveSparseRowFeatureGenerator()),
	('gen23', PositiveSparseFeatureGenerator(difficult_classes = (2, 3), extra_features = 40)),
	('gen234', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 40)),
	('gen19', PositiveSparseFeatureGenerator(difficult_classes = (1, 9), extra_features = 63)),
	('log', LogTransform()), # log should be after integer feats but before dist
	#('distp31', DistanceFeatureGenerator(n_neighbors = 3, distance_p = 1)),
	#('distp52', DistanceFeatureGenerator(n_neighbors = 5, distance_p = 2)),
	('scale03', MinMaxScaler(feature_range = (0, 3))), # scale should apply to int and float feats
]), expand_confidence = 0.94)

name = 'good01'

net = NNet.load(name = name)

probs = net.predict_proba(test)
probs = scale_to_priors(probs, priors = PRIORS)
makeSubmission(probs, fname = join(SUBMISSIONS_DIR, '{0}.csv'.format(name)), digits = 8)


