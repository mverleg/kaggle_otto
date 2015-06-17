
from os.path import join
from numpy import load, save
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from nnet.prepare import LogTransform
from nnet.scikit import NNet
from settings import SUBMISSIONS_DIR, PRIORS, BASE_DIR
from utils.features import PositiveSparseFeatureGenerator, PositiveSparseRowFeatureGenerator
from utils.ioutil import makeSubmission
from utils.loading import get_testing_data
from utils.postprocess import scale_to_priors


labels = load(join(BASE_DIR, 'data', 'trainclas.npy'))
train = load(join(BASE_DIR, 'data', 'trainmat.npy'))[:, 1:]
valid = load(join(BASE_DIR, 'data', 'testmat.npy'))[:, 1:]
test = get_testing_data()[0]

pipe = Pipeline([
	('row', PositiveSparseRowFeatureGenerator()),
	('gen23', PositiveSparseFeatureGenerator(difficult_classes = (2, 3), extra_features = 40)),
	('gen234', PositiveSparseFeatureGenerator(difficult_classes = (2, 3, 4), extra_features = 40)),
	('gen19', PositiveSparseFeatureGenerator(difficult_classes = (1, 9), extra_features = 63)),
	('log', LogTransform()), # log should be after integer feats but before dist
	#('distp31', DistanceFeatureGenerator(n_neighbors = 3, distance_p = 1)),
	#('distp52', DistanceFeatureGenerator(n_neighbors = 5, distance_p = 2)),
	('scale03', MinMaxScaler(feature_range = (0, 3))), # scale should apply to int and float feats
])

train = pipe.fit_transform(train, labels)
valid = pipe.transform(valid)
test = pipe.transform(test)

name = 'good'

net = NNet.load(name = name)

for nm, val in net.get_params().iteritems():
	print '{0:s} = {1:}'.format(nm, val)

for nm, data in [('val', valid), ('tst', test)]:
	probs = net.predict_proba(data)
	save(join(SUBMISSIONS_DIR, '{0}_{1}_raw.npy'.format(name, nm)), probs)
	makeSubmission(probs, fname = join(SUBMISSIONS_DIR, '{0}_{1}_rescale.csv'.format(name, nm)), digits = 8)
	probs = scale_to_priors(probs, priors = PRIORS)
	save(join(SUBMISSIONS_DIR, '{0}_{1}_rescale.npy'.format(name, nm)), probs)
	makeSubmission(probs, fname = join(SUBMISSIONS_DIR, '{0}_{1}_rescale.csv'.format(name, nm)), digits = 8)


