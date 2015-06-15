
from json import dumps
from sys import stdout
from sklearn.metrics import log_loss
from sklearn.metrics.scorer import _ProbaScorer


class ScoringLogging(_ProbaScorer):

	def __init__(self, score_func, sign, kwargs, log_file = None, log_treshold = None):
		self.log_file = log_file
		self.log_treshold = log_treshold
		super(ScoringLogging, self).__init__(score_func, sign, kwargs)

	def __call__(self, clf, X, y, sample_weight = None):
		"""
			Calculate the score and log to file.
		"""
		score = super(ScoringLogging, self).__call__(clf, X, y, sample_weight = sample_weight)
		params = dumps(clf.get_params())
		logtxt = '{0:.6f}\t{1:s}'.format(score, params)
		if self.log_file:
			if (self.sign > 0 and score >= self.log_treshold) \
			or (self.sign < 0 and score <= self.log_treshold):
				with open(self.log_file, 'a+') as fh:
					fh.write(logtxt)
		stdout.write('RESULT: {0:s}\n'.format(logtxt))
		return score


def get_logloss_loggingscorer(filename, treshold):
	return ScoringLogging(log_loss, sign = 1, log_file = filename, log_treshold = treshold)


