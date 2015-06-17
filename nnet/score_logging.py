
from json import dumps
from os.path import splitext
from sys import stdout
from time import strftime
from sklearn.metrics import log_loss
from sklearn.metrics.scorer import _ProbaScorer


class ScoringLogging(_ProbaScorer):

	def __init__(self, score_func, sign, kwargs, log_file = None, log_treshold = None):
		bef, aft = splitext(log_file)
		self.log_file = None
		if log_file:
			self.log_file = '{0:s}_{1:s}{2:s}'.format(bef, strftime("%Y%m%d%H%M%S"), aft)
			with open(self.log_file, 'w+') as fh:
				if log_treshold:
					fh.write('# treshold is {0:.4f}\n'.format(log_treshold))
				else:
					fh.write('# no treshold')
		self.log_treshold = log_treshold
		super(ScoringLogging, self).__init__(score_func, sign, kwargs)

	def __call__(self, clf, X, y, sample_weight = None):
		"""
			Calculate the score and log to file.
		"""
		score = super(ScoringLogging, self).__call__(clf, X, y, sample_weight = sample_weight)
		params = dumps(dict(clf.steps[-1][1].get_params()))
		logtxt = '{0:.6f}\t{1:s}'.format(score, params)
		if self.log_file:
			if (self._sign > 0 and score <= self.log_treshold) \
			or (self._sign < 0 and score >= self.log_treshold) \
			or not self.log_treshold:
				with open(self.log_file, 'a') as fh:
					fh.write(logtxt + '\n')
		stdout.write('RESULT: {0:s}\n'.format(logtxt))
		return score


def get_logloss_loggingscorer(filename, treshold):
	return ScoringLogging(log_loss, sign = -1, kwargs = {}, log_file = filename, log_treshold = treshold)


